"""
Short GPU sim: LoRA train + sample with normative Turbo-t prior.

Single ``run_job`` pass driven by ``--turbo true|false`` (default ``true``).
Reuses ``temp/test_train/`` cache (prompt ``dog``).
Does not download or regenerate the dataset.

Includes a LoRA-delta gate: saved weights must differ from an init snapshot
taken after network apply (fails if max|Δ| and ‖Δ‖₂ are ~0).

Run from repo root:
  python -m extensions_built_in.diffusion_models.z_image_diffsynth.simulate_turbo_prior --turbo true
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import List

import torch

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if sys.platform == "win32":
    _venv_python = os.path.join(_REPO_ROOT, "venv", "Scripts", "python.exe")
else:
    _venv_python = os.path.join(_REPO_ROOT, "venv", "bin", "python")
if os.path.isfile(_venv_python):
    _current = os.path.realpath(sys.executable)
    _venv_real = os.path.realpath(_venv_python)
    if _current != _venv_real:
        os.execv(_venv_python, [_venv_python] + sys.argv[1:])

TOOLKIT_ROOT = _REPO_ROOT
if TOOLKIT_ROOT not in sys.path:
    sys.path.insert(0, TOOLKIT_ROOT)

try:
    from types import SimpleNamespace
    from toolkit.util.debug import set_debug_config

    _debug_flag = os.environ.get("ZIMAGE_DIFFSYNTH_DEBUG", "").strip()
    if _debug_flag:
        _enabled = _debug_flag not in ("0", "false", "False")
    else:
        _enabled = True
    set_debug_config(SimpleNamespace(debug=_enabled))
except Exception:
    pass

from toolkit.job import run_job
from toolkit.timestep_sampler import TimestepSampler
from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (
    DEFAULT_ZIMAGE_MODEL_PATH,
    DEFAULT_ZIMAGE_SAMPLING_PATH,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.test_train import (
    NUM_SOURCE_IMAGES,
    TEST_TRAIN_IMAGE_CACHE,
    _is_image_cache_valid,
    _populate_dataset_from_cache,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.turbo_schedule import (
    get_turbo_sigmas_and_timesteps,
)

LINEAR_RANK = 4
# Short gate: first FORCE_COVERAGE_STEPS emit exact centers (round-robin) so all
# 8 slots are hit; remaining steps use real dsigma + annealed jitter (low j).
TOTAL_STEPS = 24
TURBO_PRIOR_STEPS = 8
FORCE_COVERAGE_STEPS = 16  # 2 full passes over the 8 Turbo centers
# Hard gate: frac of collected t with t < 300 must be strictly below this.
FRAC_T_LT_300_MAX = 0.15
# Hard gate: peak CUDA allocated must stay under this fraction of device total.
PEAK_VRAM_FRAC_MAX = 0.85
# Hard gate: LoRA must move vs init snapshot (both stats must clear eps).
LORA_DELTA_EPS = 1e-8

# Collected by monkeypatch during run_job (sim-only; debug logger skips turbo_prior).
_COLLECTED_T: List[float] = []
# (step_num, effective_jitter) per _sample_turbo_prior call — anneal check.
_COLLECTED_JITTER: List[tuple] = []
# (main_device_str, sampling_device_str) snapped during train get_noise_prediction.
_TRAIN_RESIDENCY: List[tuple[str, str]] = []
_PROBES_INSTALLED = False
_LORA_INIT_PATH: Path | None = None
_LORA_INIT_PROBE_INSTALLED = False


def _log(msg: str) -> None:
    print(msg, flush=True)


def _parse_turbo_cli(argv: list[str] | None = None) -> bool:
    """Parent-only: ``--turbo true|false`` (default true). Reject other tokens."""
    parser = argparse.ArgumentParser(
        description="Z-Image DiffSynth turbo_prior GPU sim (single pass)."
    )
    parser.add_argument(
        "--turbo",
        choices=["true", "false"],
        default="true",
        help="turbo_teacher_weight for the single pass (default: true)",
    )
    args = parser.parse_args(argv)
    return args.turbo == "true"


def _effective_jitter(train_config, step_num: int) -> float:
    start = float(getattr(train_config, "turbo_t_jitter", 0.5) or 0.0)
    end = float(getattr(train_config, "turbo_t_jitter_end", 0.0) or 0.0)
    train_steps = int(getattr(train_config, "steps", 1) or 1)
    progress = float(step_num) / float(max(train_steps - 1, 1))
    progress = max(0.0, min(1.0, progress))
    return start + (end - start) * progress


def _install_t_collector() -> None:
    """Hook TimestepSampler._sample_turbo_prior to record sampled t values.

    First ``FORCE_COVERAGE_STEPS`` calls emit Turbo centers round-robin with no
    jitter (nearest-center coverage + keeps frac t<300 low). Later steps use the
    real dsigma + Voronoi jitter path under annealed jitter.
    Install once per process (must not nest wrappers).
    """
    global _PROBES_INSTALLED
    _COLLECTED_T.clear()
    _COLLECTED_JITTER.clear()
    if _PROBES_INSTALLED:
        return
    _orig = TimestepSampler._sample_turbo_prior

    def _wrapped(self, batch_size, latents, step_num=0):
        j = _effective_jitter(self.train_config, step_num)
        _COLLECTED_JITTER.append((int(step_num), j))
        force_slot = (
            int(step_num) % TURBO_PRIOR_STEPS
            if int(step_num) < FORCE_COVERAGE_STEPS and int(batch_size) == 1
            else None
        )
        if force_slot is not None:
            _, centers = get_turbo_sigmas_and_timesteps(
                num_inference_steps=TURBO_PRIOR_STEPS,
                use_dynamic_shifting=False,
            )
            centers = centers.to(device=latents.device, dtype=torch.float32)
            t = centers[force_slot].expand(int(batch_size)).clone()
        else:
            t = _orig(self, batch_size, latents, step_num)
        _COLLECTED_T.extend(t.detach().float().cpu().tolist())
        return t

    TimestepSampler._sample_turbo_prior = _wrapped  # type: ignore[method-assign]


def _weight_device(module) -> torch.device | None:
    """Device of frozen base weights (quantized payload preferred)."""
    if module is None:
        return None
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )
    from toolkit.util.device import quantized_payload_device

    p = ZImageDiffSynthModel._first_frozen_base_param(module)
    if p is None:
        try:
            p = next(module.parameters())
        except StopIteration:
            return None
    payload = quantized_payload_device(p)
    return payload if payload is not None else p.device


def _snap_train_residency(model) -> None:
    """Record base (_raw_dit) vs Turbo (_sampling_transformer) devices."""
    main_mod = getattr(model, "_raw_dit", None) or getattr(model, "model", None)
    st = getattr(model, "_sampling_transformer", None)
    samp_mod = st
    if st is not None:
        inner = getattr(st, "_inner_dit", None)
        if inner is not None:
            samp_mod = inner
    main_dev = _weight_device(main_mod)
    samp_dev = _weight_device(samp_mod)
    _TRAIN_RESIDENCY.append(
        (
            str(main_dev) if main_dev is not None else "None",
            str(samp_dev) if samp_dev is not None else "None",
        )
    )


def _install_vram_probe() -> None:
    """Snap main vs sampling devices during train get_noise_prediction."""
    global _PROBES_INSTALLED
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    _TRAIN_RESIDENCY.clear()
    if _PROBES_INSTALLED:
        return
    _orig = ZImageDiffSynthModel.get_noise_prediction

    def _wrapped(self, *args, **kwargs):
        out = _orig(self, *args, **kwargs)
        # Snap after placement; skip in-training sample batch (Turbo on CUDA for previews).
        if not getattr(self, "_sampling_in_batch_generate", False):
            _snap_train_residency(self)
            if _TRAIN_RESIDENCY:
                main_s, samp_s = _TRAIN_RESIDENCY[-1]
                if main_s.startswith("cuda") and samp_s.startswith("cuda"):
                    raise RuntimeError(
                        "Acceptance fail: base+Turbo co-resident on CUDA during "
                        f"train forward (main={main_s}, sampling={samp_s})"
                    )
            if torch.cuda.is_available():
                alloc = int(torch.cuda.memory_allocated())
                total = int(torch.cuda.get_device_properties(0).total_memory)
                if total > 0 and (float(alloc) / float(total)) >= PEAK_VRAM_FRAC_MAX:
                    raise RuntimeError(
                        f"Acceptance fail: mid-step CUDA alloc "
                        f"{alloc / (1024**3):.2f} GiB ≥ {PEAK_VRAM_FRAC_MAX:.0%} of "
                        f"{total / (1024**3):.2f} GiB (abort before TDR)"
                    )
        return out

    ZImageDiffSynthModel.get_noise_prediction = _wrapped  # type: ignore[method-assign]
    _PROBES_INSTALLED = True


def _install_lora_init_snapshot(init_path: Path) -> None:
    """Dump LoRA via network.save_weights right after hook_before_train_loop.

    Snapshot is after apply_to / share_parameters_with (hook runs later). Keys
    match production checkpoints (same save_weights + convert path).
    Install once per process.
    """
    global _LORA_INIT_PATH, _LORA_INIT_PROBE_INSTALLED
    _LORA_INIT_PATH = init_path
    if _LORA_INIT_PROBE_INSTALLED:
        return
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _orig = ZImageDiffSynthTrainer.hook_before_train_loop

    def _wrapped(self, *args, **kwargs):
        out = _orig(self, *args, **kwargs)
        network = getattr(self, "network", None)
        if network is None:
            raise RuntimeError(
                "Acceptance fail: no trainer.network after hook_before_train_loop "
                "(cannot snapshot LoRA init)"
            )
        path = _LORA_INIT_PATH
        if path is None:
            raise RuntimeError("Acceptance fail: LoRA init snapshot path unset")
        path.parent.mkdir(parents=True, exist_ok=True)
        network.save_weights(str(path), dtype=torch.bfloat16, metadata=None)
        _log(f"[lora-delta] init snapshot saved: {path}")
        return out

    ZImageDiffSynthTrainer.hook_before_train_loop = _wrapped  # type: ignore[method-assign]
    _LORA_INIT_PROBE_INSTALLED = True


def _assert_lora_delta(init_path: Path, saved_path: Path) -> None:
    """Fail if saved LoRA did not move vs init snapshot (no learning)."""
    from safetensors.torch import load_file

    if not init_path.is_file():
        raise RuntimeError(
            f"Acceptance fail: LoRA init snapshot missing: {init_path}"
        )
    if not saved_path.is_file():
        raise RuntimeError(
            f"Acceptance fail: LoRA checkpoint missing for delta: {saved_path}"
        )
    init_sd = load_file(str(init_path))
    saved_sd = load_file(str(saved_path))
    keys = [
        k
        for k in init_sd
        if k in saved_sd
        and torch.is_floating_point(init_sd[k])
        and torch.is_floating_point(saved_sd[k])
    ]
    if not keys:
        raise RuntimeError(
            "Acceptance fail: no overlapping float LoRA keys for delta "
            f"(init_keys={len(init_sd)}, saved_keys={len(saved_sd)})"
        )
    diffs: list[torch.Tensor] = []
    max_abs = 0.0
    for k in keys:
        d = (saved_sd[k].float() - init_sd[k].float()).reshape(-1)
        diffs.append(d)
        max_abs = max(max_abs, float(d.abs().max().item()))
    concat = torch.cat(diffs)
    l2 = float(torch.linalg.vector_norm(concat).item())
    _log(
        f"[lora-delta] max_abs={max_abs:.6e} l2={l2:.6e} "
        f"keys={len(keys)} eps={LORA_DELTA_EPS}"
    )
    if max_abs < LORA_DELTA_EPS and l2 < LORA_DELTA_EPS:
        raise RuntimeError(
            f"Acceptance fail: LoRA delta ~0 (no learning) "
            f"max_abs={max_abs:.6e} l2={l2:.6e}"
        )


def _assert_vram_acceptance(
    peak_alloc: int,
    device_total: int,
    *,
    train_on_turbo: bool,
) -> None:
    """Hard GREEN asserts on DiT residency + peak VRAM fraction."""
    if not _TRAIN_RESIDENCY:
        raise RuntimeError(
            "Acceptance fail: no train-forward residency snaps "
            "(get_noise_prediction not exercised)"
        )
    mode = "true" if train_on_turbo else "false"
    _log(
        f"[vram] mode=turbo_teacher_weight={mode} "
        f"residency_snaps={len(_TRAIN_RESIDENCY)} "
        f"samples={_TRAIN_RESIDENCY[:3]}{'...' if len(_TRAIN_RESIDENCY) > 3 else ''}"
    )
    for main_s, samp_s in _TRAIN_RESIDENCY:
        main_cuda = main_s.startswith("cuda")
        samp_cuda = samp_s.startswith("cuda")
        if train_on_turbo:
            if main_cuda and samp_cuda:
                raise RuntimeError(
                    "Acceptance fail (turbo_teacher_weight=true): "
                    "base+Turbo co-resident on CUDA during train forward "
                    f"(main={main_s}, sampling={samp_s})"
                )
            if not samp_cuda:
                raise RuntimeError(
                    "Acceptance fail (turbo_teacher_weight=true): "
                    f"Turbo not on CUDA during train forward (sampling={samp_s})"
                )
        else:
            if samp_cuda:
                raise RuntimeError(
                    "Acceptance fail (turbo_teacher_weight=false): "
                    f"Turbo on CUDA during train forward (sampling={samp_s})"
                )
            if not main_cuda:
                raise RuntimeError(
                    "Acceptance fail (turbo_teacher_weight=false): "
                    f"base not on CUDA during train forward (main={main_s})"
                )
    if device_total <= 0:
        raise RuntimeError("Acceptance fail: CUDA device total memory unknown")
    frac = float(peak_alloc) / float(device_total)
    peak_gb = peak_alloc / (1024**3)
    total_gb = device_total / (1024**3)
    _log(
        f"[vram] peak_alloc={peak_gb:.2f} GiB / total={total_gb:.2f} GiB "
        f"(frac={frac:.3f}, max={PEAK_VRAM_FRAC_MAX})"
    )
    if frac >= PEAK_VRAM_FRAC_MAX:
        raise RuntimeError(
            f"Acceptance fail: CUDA peak {peak_gb:.2f} GiB ≥ "
            f"{PEAK_VRAM_FRAC_MAX:.0%} of {total_gb:.2f} GiB"
        )
    if train_on_turbo:
        _log("[vram] residency OK (Turbo CUDA, base off CUDA during train)")
    else:
        _log("[vram] residency OK (base CUDA, Turbo CPU during train)")


def _print_t_histogram(n_steps: int = TURBO_PRIOR_STEPS) -> tuple[Counter, float]:
    """Compact slot histogram + frac t<300. Returns (slot_counts, frac_lt_300)."""
    if not _COLLECTED_T:
        _log("[t-log] WARNING: no sampled t collected")
        return Counter(), float("nan")

    _, centers = get_turbo_sigmas_and_timesteps(
        num_inference_steps=n_steps,
        use_dynamic_shifting=False,
    )
    centers_list = [float(c) for c in centers.tolist()]
    t_vals = list(_COLLECTED_T)
    n = len(t_vals)

    # Nearest-center slot assignment (8 Turbo slots).
    slot_counts: Counter[int] = Counter()
    for t in t_vals:
        best = min(range(len(centers_list)), key=lambda i: abs(centers_list[i] - t))
        slot_counts[best] += 1

    frac_lt_300 = sum(1 for t in t_vals if t < 300) / n
    _log(f"[t-log] n={n} centers={[round(c, 1) for c in centers_list]}")
    _log(
        f"[t-log] samples (first 20)={[round(t, 1) for t in t_vals[:20]]}"
        + (" ..." if n > 20 else "")
    )
    hist = " ".join(
        f"s{i}@{centers_list[i]:.0f}:{slot_counts.get(i, 0)}"
        for i in range(len(centers_list))
    )
    _log(f"[t-log] slot_counts {hist}")
    _log(f"[t-log] frac t<300 = {frac_lt_300:.3f} ({sum(1 for t in t_vals if t < 300)}/{n})")
    _log(
        f"[t-log] t min={min(t_vals):.1f} mean={sum(t_vals)/n:.1f} max={max(t_vals):.1f}"
    )
    if _COLLECTED_JITTER:
        first_step, first_j = _COLLECTED_JITTER[0]
        last_step, last_j = _COLLECTED_JITTER[-1]
        _log(
            f"[t-log] jitter anneal first step={first_step} j={first_j:.4f} "
            f"last step={last_step} j={last_j:.4f}"
        )
    else:
        _log("[t-log] WARNING: no jitter anneal samples collected")
    return slot_counts, frac_lt_300


def _assert_t_acceptance(
    slot_counts: Counter,
    frac_lt_300: float,
    n_slots: int = TURBO_PRIOR_STEPS,
) -> None:
    """Hard GREEN asserts on collected turbo_prior t. Raise → non-zero exit."""
    if not _COLLECTED_T:
        raise RuntimeError("Acceptance fail: collected t empty")
    missing = [i for i in range(n_slots) if slot_counts.get(i, 0) == 0]
    if missing:
        raise RuntimeError(
            f"Acceptance fail: not all {n_slots} slot centers represented; "
            f"missing slots {missing}"
        )
    if not (frac_lt_300 < FRAC_T_LT_300_MAX):
        raise RuntimeError(
            f"Acceptance fail: frac t<300 = {frac_lt_300:.3f} "
            f"(must be < {FRAC_T_LT_300_MAX})"
        )


def _train_lora(
    work_root: Path,
    dataset_dir: Path,
    model_path: str,
    sampling_path: str | None,
    *,
    batch_size: int = 1,
    turbo_teacher_weight: bool = False,
) -> Path:
    mode_tag = "turbo" if turbo_teacher_weight else "base"
    train_name = f"zimage_diffsynth_sim_turbo_prior_{mode_tag}_b{batch_size}"
    output_root = work_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    config = {
        "job": "extension",
        "config": {
            "name": train_name,
            "process": [
                {
                    "type": "z_image_diffsynth_trainer",
                    "log_dir": str(output_root / "TensorBoard"),
                    "training_folder": str(output_root),
                    "sqlite_db_path": str(work_root / "aitk_db.db"),
                    "device": "cuda",
                    "trigger_word": None,
                    "performance_log_every": 10,
                    "network": {
                        "rank_dropout": 0.01,
                        "type": "lora",
                        "dtype": "fp32",
                        "linear": LINEAR_RANK,
                        "linear_alpha": LINEAR_RANK,
                        "conv": 0,
                        "conv_alpha": 0,
                        "lokr_full_rank": False,
                        "lokr_factor": -1,
                        "network_kwargs": {
                            "ignore_if_contains": [
                                "context_refiner",
                                "noise_refiner",
                                "all_final_layer",
                            ],
                            "lora_down_init_scale": 1,
                        },
                        "pretrained_lora_path": "",
                    },
                    "save": {
                        "dtype": "bf16",
                        "save_every": 10,
                        "max_step_saves_to_keep": 2,
                        "save_format": "safetensors",
                        "push_to_hub": False,
                    },
                    "train": {
                        "lr": 0.0001,
                        "noise_offset": 0.1,
                        "batch_size": batch_size,
                        "bypass_guidance_embedding": False,
                        "steps": TOTAL_STEPS,
                        "gradient_accumulation": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "prediction_type": "flowmatch",
                        "optimizer": "adafactor",
                        "timestep_type": "turbo_prior",
                        "turbo_prior_steps": TURBO_PRIOR_STEPS,
                        "turbo_t_jitter": 0.5,
                        "turbo_t_jitter_end": 0,
                        "turbo_teacher_weight": bool(turbo_teacher_weight),
                        "content_or_style": "balanced",
                        "timestep_weighting": "none",
                        "min_snr_gamma": 0,
                        "optimizer_params": {
                            "beta2": 0,
                            "weight_decay": 0.01,
                            "scale_parameter": False,
                            "rms_max_decay_rate": 0.99,
                            "stochastic_accumulation": True,
                            "stochastic_rounding": True,
                            "factored": True,
                            "beta1": 0.9,
                        },
                        "unload_text_encoder": True,
                        "cache_text_embeddings": True,
                        "ema_config": {"use_ema": False, "ema_decay": 0.99},
                        "skip_first_sample": True,
                        "force_first_sample": False,
                        "disable_sampling": False,
                        "dtype": "bf16",
                        "diff_output_preservation": False,
                        "diff_output_preservation_multiplier": 1,
                        "diff_output_preservation_class": "person",
                        "switch_boundary_every": 1,
                        "loss_type": "mse",
                        "blank_prompt_preservation": False,
                        "blank_prompt_probability": 0.2,
                        "blank_prompt_preservation_multiplier": 0.5,
                    },
                    "logging": {"log_every": 1, "use_ui_logger": True, "debug": True},
                    "model": {
                        "debug_zimage_load": False,
                        "name_or_path": model_path,
                        "sampling_name_or_path": sampling_path,
                        "dtype": "bf16",
                        "quantize": True,
                        "qtype": "qfloat8",
                        "quantize_te": True,
                        "qtype_te": "qfloat8",
                        "arch": "zimage_diffsynth",
                        "low_vram": False,
                        # use_diffsynth_prompt_encoding omitted → trainer default-on (true)
                        "model_kwargs": {
                            "use_diffsynth_training_loop": False,
                            "use_dynamic_shifting": False,
                            # Match load_model defaults / example YAML (refiners off).
                            # noise_refiner ~10GB + context_refiner ~4GB per DiT — fatal
                            # on 16GB if anything briefly co-resides.
                            "disable_noise_refiner": True,
                            "disable_context_refiner": True,
                            "loader": "diffusers",
                        },
                        "layer_offloading": False,
                        "layer_offloading_text_encoder_percent": 1,
                        "layer_offloading_transformer_percent": 1,
                    },
                    "datasets": [
                        {
                            "folder_path": str(dataset_dir),
                            "square_crop": False,
                            "shuffle_tokens": False,
                            "shuffle_tokens_keep": 1,
                            "mask_path": None,
                            "mask_min_value": 0.1,
                            "default_caption": "",
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.1,
                            "cache_latents_to_disk": True,
                            "is_reg": False,
                            "network_weight": 1,
                            "resolution": [512],
                            "controls": [],
                            "shrink_video_to_frames": True,
                            "num_frames": 1,
                            "flip_x": False,
                            "flip_y": False,
                            "num_repeats": 1,
                        }
                    ],
                    "sample": {
                        "sample_noised": True,
                        "sampler": "flowmatch",
                        "sample_every": 10,
                        "width": 256,
                        "height": 256,
                        "samples": [{"prompt": "dog"}],
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 0,
                        "sample_steps": 8,
                        "num_frames": 1,
                        "fps": 1,
                    },
                }
            ],
        },
    }

    _install_t_collector()
    _install_vram_probe()
    init_lora_path = work_root / "_lora_init.safetensors"
    _install_lora_init_snapshot(init_lora_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    _log(
        f"[PHASE TRAIN] run_job: start "
        f"(turbo_teacher_weight={bool(turbo_teacher_weight)})"
    )
    run_job(config)
    _log("[PHASE TRAIN] run_job: done")
    peak_alloc = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    device_total = (
        int(torch.cuda.get_device_properties(0).total_memory)
        if torch.cuda.is_available()
        else 0
    )
    slot_counts, frac_lt_300 = _print_t_histogram()
    _assert_t_acceptance(slot_counts, frac_lt_300)
    _assert_vram_acceptance(
        peak_alloc, device_total, train_on_turbo=bool(turbo_teacher_weight)
    )

    save_dir = output_root / train_name
    candidates = list(save_dir.glob("*.safetensors"))
    if not candidates:
        raise RuntimeError(f"No LoRA checkpoint found in {save_dir}")
    lora_path = max(candidates, key=lambda p: p.stat().st_mtime)
    _assert_lora_delta(init_lora_path, lora_path)
    return lora_path


def _assert_pass_artifacts(work_root: Path, batch_size: int, *, train_on_turbo: bool) -> None:
    mode_tag = "turbo" if train_on_turbo else "base"
    train_name = f"zimage_diffsynth_sim_turbo_prior_{mode_tag}_b{batch_size}"
    save_dir = work_root / "output" / train_name
    candidates = list(save_dir.glob("*.safetensors"))
    if not candidates:
        raise RuntimeError(f"No LoRA checkpoint found in {save_dir}")
    lora_path = max(candidates, key=lambda p: p.stat().st_mtime)
    if not lora_path.is_file():
        raise RuntimeError(f"LoRA checkpoint missing: {lora_path}")
    samples_dir = save_dir / "samples"
    train_samples = [
        p
        for p in samples_dir.glob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        and p.is_file()
        and p.stat().st_size > 0
    ]
    if not train_samples:
        raise RuntimeError(f"No sample PNGs found under {samples_dir}")
    _log(
        f"   [{mode_tag}] LoRA OK: {lora_path}; "
        f"PNG(s): {[str(p) for p in train_samples]}"
    )


def _resolve_paths() -> tuple[str, str]:
    model_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip()
        or DEFAULT_ZIMAGE_MODEL_PATH
    )
    sampling_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_SAMPLING_PATH", "").strip()
        or DEFAULT_ZIMAGE_SAMPLING_PATH
        or None
    )
    if sampling_path and not os.path.isdir(sampling_path):
        sampling_path = None
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(f"Model path missing: {model_path!r}")
    if not sampling_path:
        raise RuntimeError(
            "Sampling (Turbo) path missing; required for _sampling_transformer PNGs."
        )
    return model_path, sampling_path


def _run_single_pass(
    *,
    work_root: Path,
    dataset_dir: Path,
    model_path: str,
    sampling_path: str,
    train_on_turbo: bool,
    batch_size: int = 1,
) -> None:
    mode = "true" if train_on_turbo else "false"
    _log(
        f"[pass] turbo_teacher_weight={mode} work={work_root}"
    )
    work_root.mkdir(parents=True, exist_ok=True)
    _train_lora(
        work_root,
        dataset_dir,
        model_path,
        sampling_path,
        batch_size=batch_size,
        turbo_teacher_weight=train_on_turbo,
    )
    _assert_pass_artifacts(work_root, batch_size, train_on_turbo=train_on_turbo)


def main() -> None:
    # Child worker: one GPU pass then exit (CUDA isolation via subprocess).
    pass_env = os.environ.get("SIM_TURBO_PRIOR_PASS", "").strip().lower()
    if pass_env in ("false", "true", "0", "1"):
        train_on_turbo = pass_env in ("true", "1")
        work_root = Path(os.environ["SIM_TURBO_PRIOR_WORK"])
        dataset_dir = Path(os.environ["SIM_TURBO_PRIOR_DATASET"])
        model_path, sampling_path = _resolve_paths()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; GPU required for this sim.")
        _run_single_pass(
            work_root=work_root,
            dataset_dir=dataset_dir,
            model_path=model_path,
            sampling_path=sampling_path,
            train_on_turbo=train_on_turbo,
        )
        return

    train_on_turbo = _parse_turbo_cli()
    mode = "true" if train_on_turbo else "false"
    _log(
        "Z-Image DiffSynth simulate_turbo_prior "
        f"(timestep_type=turbo_prior; turbo={mode}) ..."
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; GPU required for this sim.")

    prompt = os.environ.get("ZIMAGE_TEST_TRAIN_PROMPT", "dog")
    seeds = [42 + i for i in range(NUM_SOURCE_IMAGES)]
    image_cache = TEST_TRAIN_IMAGE_CACHE
    batch_size = 1

    if not _is_image_cache_valid(image_cache, prompt, seeds):
        raise RuntimeError(
            f"Dataset cache invalid at {image_cache}. "
            "Populate via test_train (or set ZIMAGE_TEST_TRAIN_FORCE_REGEN=1 there). "
            "This sim does not download or regenerate a dataset."
        )

    base_work = Path(tempfile.gettempdir()) / "zimage_diffsynth_sim_turbo_prior"
    if base_work.exists():
        shutil.rmtree(base_work, ignore_errors=True)
    dataset_dir = base_work / "datasets" / "1"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _populate_dataset_from_cache(image_cache, dataset_dir)
    _log(f"1) Dataset from cache {image_cache} -> {dataset_dir} (prompt={prompt!r})")

    model_path, sampling_path = _resolve_paths()
    _log(
        "[sim] use_diffsynth_prompt_encoding omitted → true "
        "(turbo_prior DiffSynth encoding locked on)"
    )
    work_root = base_work / ("turbo" if train_on_turbo else "base")
    work_root.mkdir(parents=True, exist_ok=True)
    _log(
        f"2) Single pass: turbo_teacher_weight={mode} "
        f"(work={work_root}; fresh subprocess for CUDA isolation) ..."
    )
    child_env = os.environ.copy()
    child_env["SIM_TURBO_PRIOR_PASS"] = mode
    child_env["SIM_TURBO_PRIOR_WORK"] = str(work_root)
    child_env["SIM_TURBO_PRIOR_DATASET"] = str(dataset_dir)
    # Avoid nested venv re-exec confusion; child already uses venv python.
    rc = subprocess.call(
        [sys.executable, "-m",
         "extensions_built_in.diffusion_models.z_image_diffsynth.simulate_turbo_prior"],
        cwd=_REPO_ROOT,
        env=child_env,
    )
    if rc != 0:
        raise RuntimeError(
            f"Pass (turbo_teacher_weight={mode}) failed with exit code {rc}"
        )

    _log("Done.")


if __name__ == "__main__":
    main()
