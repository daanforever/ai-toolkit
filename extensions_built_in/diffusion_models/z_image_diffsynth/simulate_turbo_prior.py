"""
Short GPU sim: LoRA train + sample with normative Turbo-t prior.

Single ``run_job`` pass driven by ``--turbo true|false`` (default ``true``).
Reuses ``temp/test_train/`` cache (prompt ``dog``).
Does not download or regenerate the dataset.

Includes a LoRA-delta gate: saved weights must differ from an init snapshot
taken after network apply (fails if max|Δ| and ‖Δ‖₂ are ~0).

Run from repo root:
  python -m extensions_built_in.diffusion_models.z_image_diffsynth.simulate_turbo_prior --turbo true

Profiling (CUDA events + move/flush counters; sampling disabled):
  ZIMAGE_DIFFSYNTH_DEBUG=0 python -m ...simulate_turbo_prior --turbo true --profile
  ... --turbo true --profile --production-overlay   # match config.yaml recipe knobs
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

# Allow HF hub downloads for standalone te_name_or_path (override shell offline).
os.environ["HF_HUB_OFFLINE"] = "0"

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

# Standalone HF TE for verify path (root-level CausalLM; no Z-Image subfolders).
# Override with ZIMAGE_DIFFSYNTH_TE_PATH; set empty to use Z-Image snapshot TE.
DEFAULT_ZIMAGE_TE_PATH = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"

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

# Profile mode: skip first N steps, then record; sampling disabled for clean step time.
PROFILE_WARMUP_STEPS = 3
PROFILE_MEASURE_STEPS = 12
PROFILE_TOTAL_STEPS = PROFILE_WARMUP_STEPS + PROFILE_MEASURE_STEPS  # 15
PROFILE_SECTIONS = (
    "park_inactive",
    "flush_cuda",
    "move_active",
    "force_to",
    "dit_forward",
    "loss",
    "backward",
    "optimizer_step",
)

# Collected by monkeypatch during run_job (sim-only; debug logger skips turbo_prior).
_COLLECTED_T: List[float] = []
# (step_num, effective_jitter) per _sample_turbo_prior call — anneal check.
_COLLECTED_JITTER: List[tuple] = []
# (main_device_str, sampling_device_str) snapped during train get_noise_prediction.
_TRAIN_RESIDENCY: List[tuple[str, str]] = []
# True returns from _move_sampling_transformer during train forward (not sample batch).
_SAMPLING_MOVES_TRAIN: List[bool] = []
_PROBES_INSTALLED = False
_TE_CACHE_VRAM_PROBE_INSTALLED = False
# (label, alloc_gb, reserved_gb, peak_gb, total_gb, main_dev, te_dev, samp_dev)
_TE_CACHE_VRAM_EVENTS: List[tuple] = []
# Peak right after load_model (before optional reset); isolates load vs later spikes.
_PEAK_AFTER_LOAD_GB: Optional[float] = None
_LORA_INIT_PATH: Path | None = None
_LORA_INIT_PROBE_INSTALLED = False

# Profile state (sim-only; installed when --profile / SIM_TURBO_PRIOR_PROFILE).
_PROFILE_ENABLED = False
_PROFILE_PROBES_INSTALLED = False
_PROFILE_RECORDING = False  # True after warmup
_PROFILE_CPU_MS: Dict[str, List[float]] = defaultdict(list)
_PROFILE_CUDA_MS: Dict[str, List[float]] = defaultdict(list)
_PROFILE_MOVE_COUNTS: Dict[str, int] = defaultdict(int)
_PROFILE_FLUSH_COUNTS: List[int] = []  # flushes per measured step
_PROFILE_STEP_WALL_S: List[float] = []
_PROFILE_VRAM: List[tuple[float, float]] = []  # (alloc_gb, reserved_gb) per step
_PROFILE_STEP_FLUSHES = 0
_PROFILE_NEST_DEPTH = 0


def _log(msg: str) -> None:
    print(msg, flush=True)


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parent-only CLI: turbo mode + optional profile / production overlay."""
    parser = argparse.ArgumentParser(
        description="Z-Image DiffSynth turbo_prior GPU sim (single pass)."
    )
    parser.add_argument(
        "--turbo",
        choices=["true", "false"],
        default="true",
        help="turbo_teacher_weight for the single pass (default: true)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="CUDA-event step breakdown; sampling disabled; measurement-oriented",
    )
    parser.add_argument(
        "--production-overlay",
        action="store_true",
        help="Match config.yaml knobs (1024, rank 128, diffsynth, fp32, refiners on)",
    )
    return parser.parse_args(argv)


def _parse_turbo_cli(argv: list[str] | None = None) -> bool:
    """Back-compat: return turbo_teacher_weight bool from CLI."""
    return _parse_cli(argv).turbo == "true"


def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes")


def _effective_jitter(train_config, step_num: int) -> float:
    start = float(getattr(train_config, "turbo_t_jitter", 0.5) or 0.0)
    end = float(getattr(train_config, "turbo_t_jitter_end", 0.0) or 0.0)
    train_steps = int(getattr(train_config, "steps", 1) or 1)
    progress = float(step_num) / float(max(train_steps - 1, 1))
    progress = max(0.0, min(1.0, progress))
    return start + (end - start) * progress


@contextmanager
def _profile_section(name: str):
    """Record CPU wall + CUDA Event time for ``name`` when profiling after warmup."""
    if not _PROFILE_ENABLED or not _PROFILE_RECORDING:
        yield
        return
    use_cuda = torch.cuda.is_available()
    t0 = time.perf_counter()
    start_ev = end_ev = None
    if use_cuda:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
    try:
        yield
    finally:
        cpu_ms = (time.perf_counter() - t0) * 1000.0
        cuda_ms = 0.0
        if use_cuda and start_ev is not None and end_ev is not None:
            end_ev.record()
            end_ev.synchronize()
            cuda_ms = float(start_ev.elapsed_time(end_ev))
        _PROFILE_CPU_MS[name].append(cpu_ms)
        _PROFILE_CUDA_MS[name].append(cuda_ms)


def _reset_profile_buffers() -> None:
    _PROFILE_CPU_MS.clear()
    _PROFILE_CUDA_MS.clear()
    _PROFILE_MOVE_COUNTS.clear()
    _PROFILE_FLUSH_COUNTS.clear()
    _PROFILE_STEP_WALL_S.clear()
    _PROFILE_VRAM.clear()


def _median(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return float(0.5 * (s[mid - 1] + s[mid]))


def _print_profile_table() -> None:
    """Compact section table + median step wall after warmup."""
    _log("")
    _log("==== Profile Results (post-warmup) ====")
    if not _PROFILE_STEP_WALL_S:
        _log("[profile] WARNING: no measured steps recorded")
        return
    med_step = _median(_PROFILE_STEP_WALL_S)
    _log(
        f"[profile] measured_steps={len(_PROFILE_STEP_WALL_S)} "
        f"median_step_s={med_step:.4f} "
        f"mean_step_s={sum(_PROFILE_STEP_WALL_S)/len(_PROFILE_STEP_WALL_S):.4f}"
    )
    _log(
        f"{'section':<18} {'cpu_ms':>10} {'cuda_ms':>10} "
        f"{'n_samples':>10} {'moves':>8}"
    )
    section_cuda_sum = 0.0
    for name in PROFILE_SECTIONS:
        cpu_list = _PROFILE_CPU_MS.get(name, [])
        cuda_list = _PROFILE_CUDA_MS.get(name, [])
        if not cpu_list and not cuda_list:
            cpu_m = cuda_m = float("nan")
            n = 0
        else:
            cpu_m = _median(cpu_list) if cpu_list else float("nan")
            cuda_m = _median(cuda_list) if cuda_list else float("nan")
            n = max(len(cpu_list), len(cuda_list))
            if cuda_list and cuda_m == cuda_m:
                section_cuda_sum += cuda_m
        moves = int(_PROFILE_MOVE_COUNTS.get(name, 0))
        _log(
            f"{name:<18} {cpu_m:10.2f} {cuda_m:10.2f} {n:10d} {moves:8d}"
        )
    flush_med = (
        _median([float(x) for x in _PROFILE_FLUSH_COUNTS])
        if _PROFILE_FLUSH_COUNTS
        else 0.0
    )
    _log(
        f"[profile] flush_cuda calls/step median={flush_med:.1f} "
        f"(samples={len(_PROFILE_FLUSH_COUNTS)})"
    )
    if _PROFILE_VRAM:
        allocs = [a for a, _ in _PROFILE_VRAM]
        reserved = [r for _, r in _PROFILE_VRAM]
        _log(
            f"[profile] VRAM alloc_gb median={_median(allocs):.2f} "
            f"min={min(allocs):.2f} max={max(allocs):.2f}"
        )
        _log(
            f"[profile] VRAM reserved_gb median={_median(reserved):.2f} "
            f"min={min(reserved):.2f} max={max(reserved):.2f}"
        )
    step_ms = med_step * 1000.0
    if step_ms > 0:
        _log(
            f"[profile] sum(median cuda_ms sections)={section_cuda_sum:.1f} "
            f"vs median step_ms={step_ms:.1f} "
            f"(ratio={section_cuda_sum / step_ms:.2f}; nested sections overlap OK)"
        )
    move_keys = ("park_inactive", "move_active", "force_to")
    total_moves = sum(int(_PROFILE_MOVE_COUNTS.get(k, 0)) for k in move_keys)
    _log(
        "[profile] post-warmup True-move counts: "
        + ", ".join(f"{k}={int(_PROFILE_MOVE_COUNTS.get(k, 0))}" for k in move_keys)
        + f" (total={total_moves}; expect 0 after exclusive pin)"
    )
    _log("==== End Profile ====")
    _log("")


def _install_profile_probes() -> None:
    """CUDA-event + counter hooks for residency / forward / loss / backward / opt."""
    global _PROFILE_PROBES_INSTALLED, _PROFILE_RECORDING, _PROFILE_STEP_FLUSHES
    if _PROFILE_PROBES_INSTALLED:
        return
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    _reset_profile_buffers()
    _PROFILE_RECORDING = False
    _PROFILE_STEP_FLUSHES = 0

    _orig_place = ZImageDiffSynthModel._place_training_dit
    _orig_move_samp = ZImageDiffSynthModel._move_sampling_transformer
    _orig_move_main = ZImageDiffSynthModel._move_main_network
    _orig_flush = ZImageDiffSynthModel._flush_cuda
    _orig_force = ZImageDiffSynthModel._force_network_to
    _orig_gnp = ZImageDiffSynthModel.get_noise_prediction
    _orig_calc_loss = SDTrainer.calculate_loss
    _orig_hook = SDTrainer.hook_train_loop

    def _place(self, device):
        target = device if isinstance(device, torch.device) else torch.device(device)
        section = "park_inactive" if target.type == "cpu" else "move_active"
        with _profile_section(section):
            moved = _orig_place(self, device)
        if _PROFILE_RECORDING and moved:
            _PROFILE_MOVE_COUNTS[section] += 1
        return moved

    def _move_samp(self, device):
        target = device if isinstance(device, torch.device) else torch.device(device)
        section = "park_inactive" if target.type == "cpu" else "move_active"
        with _profile_section(section):
            moved = _orig_move_samp(self, device)
        if (
            _PROFILE_RECORDING
            and moved
            and not getattr(self, "_sampling_in_batch_generate", False)
        ):
            _PROFILE_MOVE_COUNTS[section] += 1
        return moved

    def _move_main(self, device):
        with _profile_section("move_active"):
            return _orig_move_main(self, device)

    def _flush(self):
        global _PROFILE_STEP_FLUSHES
        with _profile_section("flush_cuda"):
            out = _orig_flush(self)
        if _PROFILE_RECORDING:
            _PROFILE_STEP_FLUSHES += 1
            _PROFILE_MOVE_COUNTS["flush_cuda"] += 1
        return out

    def _force(self, net, device):
        with _profile_section("force_to"):
            out = _orig_force(self, net, device)
        if _PROFILE_RECORDING:
            _PROFILE_MOVE_COUNTS["force_to"] += 1
        return out

    def _gnp(self, *args, **kwargs):
        with _profile_section("dit_forward"):
            return _orig_gnp(self, *args, **kwargs)

    def _calc_loss(self, *args, **kwargs):
        with _profile_section("loss"):
            return _orig_calc_loss(self, *args, **kwargs)

    def _hook_train_loop(self, batch):
        global _PROFILE_RECORDING, _PROFILE_STEP_FLUSHES
        step = int(getattr(self, "step_num", 0) or 0)
        if _PROFILE_ENABLED and step >= PROFILE_WARMUP_STEPS:
            _PROFILE_RECORDING = True
        else:
            _PROFILE_RECORDING = False
        _PROFILE_STEP_FLUSHES = 0
        t0 = time.perf_counter()
        accel = getattr(self, "accelerator", None)
        orig_backward = getattr(accel, "backward", None) if accel is not None else None
        opt = getattr(self, "optimizer", None)
        orig_step = getattr(opt, "step", None) if opt is not None else None
        if orig_backward is not None and _PROFILE_RECORDING:

            def _backward(loss, *a, **kw):
                with _profile_section("backward"):
                    return orig_backward(loss, *a, **kw)

            accel.backward = _backward  # type: ignore[method-assign]
        if orig_step is not None and _PROFILE_RECORDING:

            def _opt_step(*a, **kw):
                with _profile_section("optimizer_step"):
                    return orig_step(*a, **kw)

            opt.step = _opt_step  # type: ignore[method-assign]
        try:
            return _orig_hook(self, batch)
        finally:
            if orig_backward is not None and accel is not None:
                accel.backward = orig_backward  # type: ignore[method-assign]
            if orig_step is not None and opt is not None:
                opt.step = orig_step  # type: ignore[method-assign]
            if _PROFILE_RECORDING:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                _PROFILE_STEP_WALL_S.append(wall)
                _PROFILE_FLUSH_COUNTS.append(_PROFILE_STEP_FLUSHES)
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    _PROFILE_VRAM.append((float(alloc), float(reserved)))
                    _log(
                        f"[profile] step={step} wall_s={wall:.4f} "
                        f"flush={_PROFILE_STEP_FLUSHES} "
                        f"alloc_gb={alloc:.2f} reserved_gb={reserved:.2f}"
                    )
                else:
                    _log(
                        f"[profile] step={step} wall_s={wall:.4f} "
                        f"flush={_PROFILE_STEP_FLUSHES}"
                    )

    ZImageDiffSynthModel._place_training_dit = _place  # type: ignore[method-assign]
    ZImageDiffSynthModel._move_sampling_transformer = _move_samp  # type: ignore[method-assign]
    ZImageDiffSynthModel._move_main_network = _move_main  # type: ignore[method-assign]
    ZImageDiffSynthModel._flush_cuda = _flush  # type: ignore[method-assign]
    ZImageDiffSynthModel._force_network_to = _force  # type: ignore[method-assign]
    ZImageDiffSynthModel.get_noise_prediction = _gnp  # type: ignore[method-assign]
    SDTrainer.calculate_loss = _calc_loss  # type: ignore[method-assign]
    SDTrainer.hook_train_loop = _hook_train_loop  # type: ignore[method-assign]
    _PROFILE_PROBES_INSTALLED = True
    _log(
        f"[profile] probes installed (warmup={PROFILE_WARMUP_STEPS}, "
        f"measure≈{PROFILE_MEASURE_STEPS})"
    )


def _install_t_collector() -> None:
    """Hook TimestepSampler._sample_turbo_prior to record sampled t values.

    First ``FORCE_COVERAGE_STEPS`` calls emit Turbo centers round-robin with no
    jitter (nearest-center coverage + keeps frac t<300 low). Later steps use the
    real dsigma + Voronoi jitter path under annealed jitter
    (content may reverse dsigma).
    Install once per process (must not nest wrappers).
    """
    global _PROBES_INSTALLED
    _COLLECTED_T.clear()
    _COLLECTED_JITTER.clear()
    if _PROBES_INSTALLED:
        return
    _orig = TimestepSampler._sample_turbo_prior

    def _wrapped(self, batch_size, latents, step_num=0, content_or_style="balanced"):
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
            t = _orig(self, batch_size, latents, step_num, content_or_style)
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


def _te_weight_device(model) -> str:
    te = getattr(model, "text_encoder", None)
    if te is None:
        return "None"
    if isinstance(te, list):
        te = te[0] if te else None
    if te is None:
        return "None"
    # FakeTextEncoder has .device property but no real weights on CUDA.
    try:
        from toolkit.unloader import FakeTextEncoder

        if isinstance(te, FakeTextEncoder):
            return f"fake:{te.device}"
    except Exception:
        pass
    d = _weight_device(te)
    return str(d) if d is not None else "None"


def _cuda_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _log_te_cache_vram(label: str, model=None) -> None:
    """Append + print CUDA alloc/peak/reserved and DiT/TE/Turbo devices."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    total = _cuda_total_gb()
    frac = (peak / total) if total > 0 else 0.0
    gate = "FAIL" if total > 0 and frac >= PEAK_VRAM_FRAC_MAX else "ok"
    main_s = te_s = samp_s = "?"
    if model is not None:
        main_mod = getattr(model, "_raw_dit", None) or getattr(model, "model", None)
        st = getattr(model, "_sampling_transformer", None)
        samp_mod = getattr(st, "_inner_dit", None) if st is not None else None
        if samp_mod is None:
            samp_mod = st
        md = _weight_device(main_mod)
        sd = _weight_device(samp_mod)
        main_s = str(md) if md is not None else "None"
        samp_s = str(sd) if sd is not None else "None"
        te_s = _te_weight_device(model)
    evt = (label, alloc, reserved, peak, total, main_s, te_s, samp_s)
    _TE_CACHE_VRAM_EVENTS.append(evt)
    _log(
        f"[te-cache-vram] {label}: alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB "
        f"peak={peak:.2f}/{total:.2f}GiB ({frac:.1%} {gate}) "
        f"main={main_s} te={te_s} samp={samp_s}"
    )


def _print_te_cache_vram_timeline() -> None:
    if not _TE_CACHE_VRAM_EVENTS:
        _log("[te-cache-vram] no events recorded")
        return
    _log("[te-cache-vram] === timeline ===")
    if _PEAK_AFTER_LOAD_GB is not None:
        _log(
            f"  (load sticky peak={_PEAK_AFTER_LOAD_GB:.2f}GiB; "
            "later Δpeak above this is post-load)"
        )
    prev_peak = 0.0
    for label, alloc, reserved, peak, total, main_s, te_s, samp_s in _TE_CACHE_VRAM_EVENTS:
        dpeak = peak - prev_peak
        mark = " <<" if dpeak > 0.05 else ""
        frac = (peak / total) if total > 0 else 0.0
        gate = "FAIL" if total > 0 and frac >= PEAK_VRAM_FRAC_MAX else "ok"
        _log(
            f"  {label}: alloc={alloc:.2f} reserved={reserved:.2f} peak={peak:.2f} "
            f"(Δpeak={dpeak:+.2f}, {frac:.1%} {gate}) "
            f"main={main_s} te={te_s} samp={samp_s}{mark}"
        )
        prev_peak = peak
    _log("[te-cache-vram] === end ===")


def _install_te_cache_vram_probe() -> None:
    """Monkeypatch enter/exit/unload/cache/load/turbo/sample for VRAM timeline."""
    global _TE_CACHE_VRAM_PROBE_INSTALLED, _PEAK_AFTER_LOAD_GB
    _TE_CACHE_VRAM_EVENTS.clear()
    _PEAK_AFTER_LOAD_GB = None
    if _TE_CACHE_VRAM_PROBE_INSTALLED:
        return

    import toolkit.unloader as unloader_mod
    from toolkit.dataloader_mixins import LatentCachingMixin, TextEmbeddingCachingMixin
    from toolkit.network_mixins import ToolkitNetworkMixin
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    _orig_enter = unloader_mod.enter_text_cache_residency
    _orig_exit = unloader_mod.exit_text_cache_residency
    _orig_abort = unloader_mod.abort_text_cache_residency
    _orig_unload = unloader_mod.unload_text_encoder
    _orig_cache = TextEmbeddingCachingMixin.cache_text_embeddings
    _orig_cache_latents = LatentCachingMixin.cache_latents_all_latents
    _orig_force_to = ToolkitNetworkMixin.force_to
    _orig_load = ZImageDiffSynthModel.load_model
    _orig_turbo = ZImageDiffSynthModel.apply_turbo_teacher_mode
    _orig_gen = ZImageDiffSynthModel.generate_images
    _orig_set_state = ZImageDiffSynthModel.set_device_state
    _orig_encode = ZImageDiffSynthModel.encode_prompt
    _encode_call_n = {"n": 0}

    def _enter(model, device=None):
        _log_te_cache_vram("enter:before", model)
        out = _orig_enter(model, device)
        _log_te_cache_vram("enter:after", model)
        return out

    def _exit(model, device=None):
        _log_te_cache_vram("exit:before", model)
        out = _orig_exit(model, device)
        _log_te_cache_vram("exit:after", model)
        return out

    def _abort(model):
        _log_te_cache_vram("abort:before", model)
        out = _orig_abort(model)
        _log_te_cache_vram("abort:after", model)
        return out

    def _unload(model):
        _log_te_cache_vram("unload_te:before", model)
        out = _orig_unload(model)
        _log_te_cache_vram("unload_te:after", model)
        return out

    def _cache(self):
        sd = getattr(self, "sd", None)
        _encode_call_n["n"] = 0
        _log_te_cache_vram("cache_text_embeddings:enter", sd)
        try:
            return _orig_cache(self)
        finally:
            _log_te_cache_vram("cache_text_embeddings:exit", sd)

    def _cache_latents(self):
        sd = getattr(self, "sd", None)
        _log_te_cache_vram("cache_latents:enter", sd)
        try:
            return _orig_cache_latents(self)
        finally:
            _log_te_cache_vram("cache_latents:exit", sd)

    def _force_to(self, device, dtype):
        sd = None
        ref = getattr(self, "base_model_ref", None)
        if ref is not None:
            try:
                sd = ref()
            except Exception:
                sd = None
        _log_te_cache_vram(f"lora.force_to:before({device})", sd)
        out = _orig_force_to(self, device, dtype)
        _log_te_cache_vram(f"lora.force_to:after({device})", sd)
        return out

    def _load(self):
        _log_te_cache_vram("load_model:enter", self)
        out = _orig_load(self)
        _log_te_cache_vram("load_model:exit", self)
        global _PEAK_AFTER_LOAD_GB
        if torch.cuda.is_available():
            _PEAK_AFTER_LOAD_GB = torch.cuda.max_memory_allocated() / (1024**3)
            # Isolate post-load spikes (TE cache / train / sample) from sticky load peak.
            if _env_flag("ZIMAGE_SIM_RESET_PEAK_AFTER_LOAD"):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                _log(
                    f"[te-cache-vram] reset_peak after load "
                    f"(saved sticky={_PEAK_AFTER_LOAD_GB:.2f}GiB)"
                )
                _log_te_cache_vram("load_model:after_peak_reset", self)
        return out

    def _turbo(self, enabled: bool):
        _log_te_cache_vram(f"apply_turbo:before(enabled={bool(enabled)})", self)
        out = _orig_turbo(self, enabled)
        _log_te_cache_vram(f"apply_turbo:after(enabled={bool(enabled)})", self)
        return out

    def _gen(self, *args, **kwargs):
        _log_te_cache_vram("generate_images:enter", self)
        try:
            return _orig_gen(self, *args, **kwargs)
        finally:
            _log_te_cache_vram("generate_images:exit", self)

    def _set_state(self, state):
        _log_te_cache_vram("set_device_state:enter", self)
        out = _orig_set_state(self, state)
        _log_te_cache_vram("set_device_state:exit", self)
        return out

    def _encode(self, *args, **kwargs):
        # Log first two encodes during text-embed cache (activation spike source).
        n = _encode_call_n["n"]
        _encode_call_n["n"] = n + 1
        if n < 2:
            _log_te_cache_vram(f"encode_prompt:before#{n}", self)
        out = _orig_encode(self, *args, **kwargs)
        if n < 2:
            _log_te_cache_vram(f"encode_prompt:after#{n}", self)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _log_te_cache_vram(f"encode_prompt:after_flush#{n}", self)
        return out

    setattr(unloader_mod, "enter_text_cache_residency", _enter)
    setattr(unloader_mod, "exit_text_cache_residency", _exit)
    setattr(unloader_mod, "abort_text_cache_residency", _abort)
    setattr(unloader_mod, "unload_text_encoder", _unload)
    TextEmbeddingCachingMixin.cache_text_embeddings = _cache  # type: ignore[method-assign]
    LatentCachingMixin.cache_latents_all_latents = _cache_latents  # type: ignore[method-assign]
    ToolkitNetworkMixin.force_to = _force_to  # type: ignore[method-assign]
    ZImageDiffSynthModel.load_model = _load  # type: ignore[method-assign]
    ZImageDiffSynthModel.apply_turbo_teacher_mode = _turbo  # type: ignore[method-assign]
    ZImageDiffSynthModel.generate_images = _gen  # type: ignore[method-assign]
    ZImageDiffSynthModel.set_device_state = _set_state  # type: ignore[method-assign]
    ZImageDiffSynthModel.encode_prompt = _encode  # type: ignore[method-assign]

    # Re-bind imports used by call sites that already imported symbols.
    import toolkit.dataloader_mixins as dlm
    import extensions_built_in.sd_trainer.SDTrainer as sdt

    setattr(dlm, "enter_text_cache_residency", _enter)
    setattr(dlm, "abort_text_cache_residency", _abort)
    setattr(sdt, "enter_text_cache_residency", _enter)
    setattr(sdt, "exit_text_cache_residency", _exit)
    setattr(sdt, "abort_text_cache_residency", _abort)
    setattr(sdt, "unload_text_encoder", _unload)

    _TE_CACHE_VRAM_PROBE_INSTALLED = True
    _log(
        "[te-cache-vram] probe installed "
        f"(reset_peak_after_load={_env_flag('ZIMAGE_SIM_RESET_PEAK_AFTER_LOAD')})"
    )


def _install_vram_probe() -> None:
    """Snap main vs sampling devices during train get_noise_prediction.

    Also counts real ``_move_sampling_transformer`` moves (True returns) during
    train forward; sample-batch moves (``_sampling_in_batch_generate``) ignored.
    """
    global _PROBES_INSTALLED
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    _TRAIN_RESIDENCY.clear()
    _SAMPLING_MOVES_TRAIN.clear()
    if _PROBES_INSTALLED:
        return
    _orig = ZImageDiffSynthModel.get_noise_prediction
    _orig_move = ZImageDiffSynthModel._move_sampling_transformer

    def _wrapped_move(self, device):
        moved = _orig_move(self, device)
        if moved and not getattr(self, "_sampling_in_batch_generate", False):
            _SAMPLING_MOVES_TRAIN.append(True)
        return moved

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

    ZImageDiffSynthModel._move_sampling_transformer = _wrapped_move  # type: ignore[method-assign]
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
    if _PEAK_AFTER_LOAD_GB is not None:
        _log(
            f"[vram] sticky_peak_after_load={_PEAK_AFTER_LOAD_GB:.2f} GiB "
            f"(post-load contribution≈{max(0.0, peak_gb - _PEAK_AFTER_LOAD_GB):.2f} GiB)"
        )
    if frac >= PEAK_VRAM_FRAC_MAX:
        raise RuntimeError(
            f"Acceptance fail: CUDA peak {peak_gb:.2f} GiB ≥ "
            f"{PEAK_VRAM_FRAC_MAX:.0%} of {total_gb:.2f} GiB"
            + (
                f" (sticky load peak was {_PEAK_AFTER_LOAD_GB:.2f} GiB)"
                if _PEAK_AFTER_LOAD_GB is not None
                else ""
            )
        )
    n_moves = len(_SAMPLING_MOVES_TRAIN)
    first_allowed = 1
    extras = max(0, n_moves - first_allowed)
    _log(
        f"[vram] sampling_moves_train={n_moves} "
        f"first_allowed={first_allowed} extras={extras}"
    )
    if extras > 0:
        raise RuntimeError(
            "Acceptance fail: sampling transformer moved during train forward "
            "after exclusive pin"
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
    te_name_or_path: str | None = None,
    batch_size: int = 1,
    turbo_teacher_weight: bool = False,
    profile: bool = False,
    production_overlay: bool = False,
) -> Path:
    global _PROFILE_ENABLED, FORCE_COVERAGE_STEPS
    mode_tag = "turbo" if turbo_teacher_weight else "base"
    train_name = f"zimage_diffsynth_sim_turbo_prior_{mode_tag}_b{batch_size}"
    output_root = work_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    n_steps = PROFILE_TOTAL_STEPS if profile else TOTAL_STEPS
    lora_rank = 128 if production_overlay else LINEAR_RANK
    resolution = [1024] if production_overlay else [512]
    train_dtype = "fp32" if production_overlay else "bf16"
    loader = "diffsynth" if production_overlay else "diffusers"
    # Production config.yaml leaves refiners enabled; sim default is off (VRAM).
    disable_refiners = not production_overlay
    disable_sampling = bool(profile)
    skip_first_sample = True
    save_every = max(n_steps, 1) if profile else 10
    force_coverage = min(FORCE_COVERAGE_STEPS, max(0, n_steps - 1))

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
                    "performance_log_every": 10 if not profile else 5,
                    "network": {
                        "rank_dropout": 0.01,
                        "type": "lora",
                        "dtype": "fp32",
                        "linear": lora_rank,
                        "linear_alpha": lora_rank,
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
                        "save_every": save_every,
                        "max_step_saves_to_keep": 2,
                        "save_format": "safetensors",
                        "push_to_hub": False,
                    },
                    "train": {
                        "lr": 0.0001,
                        "noise_offset": 0.1,
                        "batch_size": batch_size,
                        "bypass_guidance_embedding": False,
                        "steps": n_steps,
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
                        "skip_first_sample": skip_first_sample,
                        "force_first_sample": False,
                        "disable_sampling": disable_sampling,
                        "dtype": train_dtype,
                        "diff_output_preservation": False,
                        "diff_output_preservation_multiplier": 1,
                        "diff_output_preservation_class": "person",
                        "switch_boundary_every": 1,
                        "loss_type": "mse",
                        "blank_prompt_preservation": False,
                        "blank_prompt_probability": 0.2,
                        "blank_prompt_preservation_multiplier": 0.5,
                    },
                    "logging": {
                        "log_every": 1,
                        "use_ui_logger": True,
                        "debug": not profile,
                    },
                    "model": {
                        "debug_zimage_load": False,
                        "name_or_path": model_path,
                        "sampling_name_or_path": sampling_path,
                        "te_name_or_path": te_name_or_path,
                        "dtype": "bf16",
                        "quantize": True,
                        "qtype": "qfloat8",
                        "quantize_te": False,
                        "qtype_te": "qfloat8",
                        "arch": "zimage_diffsynth",
                        "low_vram": False,
                        # use_diffsynth_prompt_encoding omitted → trainer default-on (true)
                        "model_kwargs": {
                            "use_diffsynth_training_loop": False,
                            "use_dynamic_shifting": False,
                            # noise_refiner ~10GB + context_refiner ~4GB per DiT — fatal
                            # on 16GB if anything briefly co-resides (sim default: off).
                            "disable_noise_refiner": disable_refiners,
                            "disable_context_refiner": disable_refiners,
                            "loader": loader,
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
                            "resolution": resolution,
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
                        "sample_every": 10 if not profile else 10_000,
                        "width": 256 if not production_overlay else 1024,
                        "height": 256 if not production_overlay else 768,
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

    # Shorter force-coverage window when profiling with fewer steps.
    _saved_force = FORCE_COVERAGE_STEPS
    if profile:
        FORCE_COVERAGE_STEPS = force_coverage

    _PROFILE_ENABLED = bool(profile)
    _install_t_collector()
    _install_vram_probe()
    _install_te_cache_vram_probe()
    if profile:
        _install_profile_probes()
    init_lora_path = work_root / "_lora_init.safetensors"
    _install_lora_init_snapshot(init_lora_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    _log(
        f"[PHASE TRAIN] run_job: start "
        f"(turbo_teacher_weight={bool(turbo_teacher_weight)} "
        f"profile={bool(profile)} production_overlay={bool(production_overlay)} "
        f"steps={n_steps} rank={lora_rank} res={resolution} "
        f"dtype={train_dtype} loader={loader} refiners_off={disable_refiners})"
    )
    try:
        run_job(config)
    finally:
        FORCE_COVERAGE_STEPS = _saved_force
        _print_te_cache_vram_timeline()
    _log("[PHASE TRAIN] run_job: done")
    if profile:
        _print_profile_table()
    peak_alloc = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    device_total = (
        int(torch.cuda.get_device_properties(0).total_memory)
        if torch.cuda.is_available()
        else 0
    )
    slot_counts, frac_lt_300 = _print_t_histogram()
    # Profile with fewer steps may miss full 8-slot coverage; keep VRAM/residency.
    if not profile:
        _assert_t_acceptance(slot_counts, frac_lt_300)
    else:
        _log(
            "[profile] skipping full t-slot acceptance "
            f"(force_coverage={force_coverage}, steps={n_steps})"
        )
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


def _assert_pass_artifacts(
    work_root: Path,
    batch_size: int,
    *,
    train_on_turbo: bool,
    profile: bool = False,
) -> None:
    mode_tag = "turbo" if train_on_turbo else "base"
    train_name = f"zimage_diffsynth_sim_turbo_prior_{mode_tag}_b{batch_size}"
    save_dir = work_root / "output" / train_name
    candidates = list(save_dir.glob("*.safetensors"))
    if not candidates:
        raise RuntimeError(f"No LoRA checkpoint found in {save_dir}")
    lora_path = max(candidates, key=lambda p: p.stat().st_mtime)
    if not lora_path.is_file():
        raise RuntimeError(f"LoRA checkpoint missing: {lora_path}")
    if profile:
        _log(
            f"   [{mode_tag}] LoRA OK: {lora_path}; "
            "sample PNGs skipped (--profile disables sampling)"
        )
        return
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


def _resolve_paths() -> tuple[str, str, str | None]:
    model_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip()
        or DEFAULT_ZIMAGE_MODEL_PATH
    )
    sampling_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_SAMPLING_PATH", "").strip()
        or DEFAULT_ZIMAGE_SAMPLING_PATH
        or None
    )
    # Unset → DEFAULT_ZIMAGE_TE_PATH; empty string → stock Z-Image snapshot TE.
    if "ZIMAGE_DIFFSYNTH_TE_PATH" in os.environ:
        te_path = os.environ.get("ZIMAGE_DIFFSYNTH_TE_PATH", "").strip() or None
    else:
        te_path = DEFAULT_ZIMAGE_TE_PATH
    if sampling_path and not os.path.isdir(sampling_path):
        sampling_path = None
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(f"Model path missing: {model_path!r}")
    if not sampling_path:
        raise RuntimeError(
            "Sampling (Turbo) path missing; required for _sampling_transformer PNGs."
        )
    return model_path, sampling_path, te_path


def _run_single_pass(
    *,
    work_root: Path,
    dataset_dir: Path,
    model_path: str,
    sampling_path: str,
    te_name_or_path: str | None = None,
    train_on_turbo: bool,
    batch_size: int = 1,
    profile: bool = False,
    production_overlay: bool = False,
) -> None:
    mode = "true" if train_on_turbo else "false"
    _log(
        f"[pass] turbo_teacher_weight={mode} work={work_root} "
        f"profile={profile} production_overlay={production_overlay}"
    )
    work_root.mkdir(parents=True, exist_ok=True)
    _train_lora(
        work_root,
        dataset_dir,
        model_path,
        sampling_path,
        te_name_or_path=te_name_or_path,
        batch_size=batch_size,
        turbo_teacher_weight=train_on_turbo,
        profile=profile,
        production_overlay=production_overlay,
    )
    _assert_pass_artifacts(
        work_root, batch_size, train_on_turbo=train_on_turbo, profile=profile
    )


def main() -> None:
    # Child worker: one GPU pass then exit (CUDA isolation via subprocess).
    pass_env = os.environ.get("SIM_TURBO_PRIOR_PASS", "").strip().lower()
    if pass_env in ("false", "true", "0", "1"):
        train_on_turbo = pass_env in ("true", "1")
        work_root = Path(os.environ["SIM_TURBO_PRIOR_WORK"])
        dataset_dir = Path(os.environ["SIM_TURBO_PRIOR_DATASET"])
        profile = _env_flag("SIM_TURBO_PRIOR_PROFILE")
        production_overlay = _env_flag("SIM_TURBO_PRIOR_PRODUCTION")
        model_path, sampling_path, te_path = _resolve_paths()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; GPU required for this sim.")
        _run_single_pass(
            work_root=work_root,
            dataset_dir=dataset_dir,
            model_path=model_path,
            sampling_path=sampling_path,
            te_name_or_path=te_path,
            train_on_turbo=train_on_turbo,
            profile=profile,
            production_overlay=production_overlay,
        )
        return

    args = _parse_cli()
    train_on_turbo = args.turbo == "true"
    profile = bool(args.profile)
    production_overlay = bool(args.production_overlay)
    mode = "true" if train_on_turbo else "false"
    _log(
        "Z-Image DiffSynth simulate_turbo_prior "
        f"(timestep_type=turbo_prior; turbo={mode}; "
        f"profile={profile}; production_overlay={production_overlay}) ..."
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

    model_path, sampling_path, te_path = _resolve_paths()
    _log(
        "[sim] use_diffsynth_prompt_encoding omitted → true "
        "(turbo_prior DiffSynth encoding locked on)"
    )
    if te_path:
        _log(f"[sim] te_name_or_path={te_path!r}")
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
    if profile:
        child_env["SIM_TURBO_PRIOR_PROFILE"] = "1"
    if production_overlay:
        child_env["SIM_TURBO_PRIOR_PRODUCTION"] = "1"
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
