"""
Short GPU sim: LoRA train + sample with normative Turbo-t prior.

One process, one ``run_job``, one model load. Reuses ``temp/test_train/`` cache
(prompt ``dog``). Does not download or regenerate the dataset.

Run from repo root:
  python -m extensions_built_in.diffusion_models.z_image_diffsynth.simulate_turbo_prior
"""

from __future__ import annotations

import os
import shutil
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

# Collected by monkeypatch during run_job (sim-only; debug logger skips turbo_prior).
_COLLECTED_T: List[float] = []
# (step_num, effective_jitter) per _sample_turbo_prior call — anneal check.
_COLLECTED_JITTER: List[tuple] = []


def _log(msg: str) -> None:
    print(msg, flush=True)


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
    """
    _COLLECTED_T.clear()
    _COLLECTED_JITTER.clear()
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
) -> Path:
    train_name = f"zimage_diffsynth_sim_turbo_prior_b{batch_size}"
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
                        "turbo_teacher_weight": 0.25,
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
                            "disable_noise_refiner": False,
                            "disable_context_refiner": False,
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
    _log("[PHASE TRAIN] run_job: start")
    run_job(config)
    _log("[PHASE TRAIN] run_job: done")
    slot_counts, frac_lt_300 = _print_t_histogram()
    _assert_t_acceptance(slot_counts, frac_lt_300)

    save_dir = output_root / train_name
    candidates = list(save_dir.glob("*.safetensors"))
    if not candidates:
        raise RuntimeError(f"No LoRA checkpoint found in {save_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    _log("Z-Image DiffSynth simulate_turbo_prior (timestep_type=turbo_prior) ...")
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

    work_root = Path(tempfile.gettempdir()) / "zimage_diffsynth_sim_turbo_prior"
    if work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)
    dataset_dir = work_root / "datasets" / "1"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _populate_dataset_from_cache(image_cache, dataset_dir)
    _log(f"1) Dataset from cache {image_cache} -> {dataset_dir} (prompt={prompt!r})")

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

    _log("2) Training LoRA + sample via single run_job (turbo_prior) ...")
    _log(
        "[sim] use_diffsynth_prompt_encoding omitted → true "
        "(turbo_prior DiffSynth encoding locked on)"
    )
    lora_path = _train_lora(
        work_root, dataset_dir, model_path, sampling_path, batch_size=batch_size
    )
    if not lora_path.is_file():
        raise RuntimeError(f"LoRA checkpoint missing: {lora_path}")
    _log(f"   LoRA checkpoint: {lora_path}")

    samples_dir = (
        work_root / "output" / f"zimage_diffsynth_sim_turbo_prior_b{batch_size}" / "samples"
    )
    train_samples = [
        p
        for p in samples_dir.glob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        and p.is_file()
        and p.stat().st_size > 0
    ]
    if not train_samples:
        raise RuntimeError(f"No sample PNGs found under {samples_dir}")

    _log(f"3) Acceptance PNG(s): {[str(p) for p in train_samples]}")
    _log(f"   LoRA OK: {lora_path}")
    _log("Done. ACCEPTANCE GREEN")


if __name__ == "__main__":
    main()
