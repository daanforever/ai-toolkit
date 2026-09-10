"""
Short GPU sim: LoRA train + sample with normative Turbo-t prior.

Single ``run_job`` pass driven by ``simulate_turbo_prior.yaml`` (no CLI).
Reuses ``temp/test_train/`` cache (prompt ``dog``).
Does not download or regenerate the dataset.

Includes a LoRA-delta gate: saved weights must differ from an init snapshot
taken after network apply (fails if max|Δ| and ‖Δ‖₂ are ~0).

Run from repo root:
  python -m extensions_built_in.diffusion_models.z_image_diffsynth.simulate_turbo_prior

Edit ``simulate_turbo_prior.yaml`` for knobs (quantize, steps, refiners, sample size).
Set ``sim.profile: true`` for CUDA-event step breakdown (sampling disabled).
"""

from __future__ import annotations

import copy
import faulthandler
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

import yaml

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
from toolkit.train_tools import get_torch_dtype
from extensions_built_in.diffusion_models.z_image_diffsynth.test_train import (
    NUM_SOURCE_IMAGES,
    TEST_TRAIN_IMAGE_CACHE,
    _is_image_cache_valid,
    _populate_dataset_from_cache,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.turbo_schedule import (
    get_turbo_sigmas_and_timesteps,
)

# Short gate: first FORCE_COVERAGE_STEPS emit exact centers (round-robin) so all
# 8 slots are hit; remaining steps use real dsigma + annealed jitter (low j).
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

DEFAULT_YAML_PATH = Path(__file__).with_name("simulate_turbo_prior.yaml")

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
# (label, alloc_gb, reserved_gb, peak_gb, total_gb, main_dev, te_dev, samp_dev, vae_dev)
_TE_CACHE_VRAM_EVENTS: List[tuple] = []
# Peak right after load_model (before optional reset); isolates load vs later spikes.
_PEAK_AFTER_LOAD_GB: Optional[float] = None
_LORA_INIT_PATH: Path | None = None
_LORA_INIT_PROBE_INSTALLED = False

# Profile state (sim-only; installed when sim.profile is true).
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
# When False, peak-VRAM ≥ max is warning-only (unquantized DiT).
_QUANTIZE_HARD_PEAK = True


def _log(msg: str) -> None:
    print(msg, flush=True)


def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes")


def _default_yaml_path() -> Path:
    return DEFAULT_YAML_PATH


def _load_recipe(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"recipe at {path} must be a mapping")
    return data


def _parse_sim(process0: dict) -> dict:
    sim = copy.deepcopy(process0.get("sim") or {})
    if not isinstance(sim, dict):
        raise ValueError("'sim' must be a mapping")
    return {"profile": bool(sim.get("profile", False))}


def _strip_sim(config: dict) -> dict:
    out = copy.deepcopy(config)
    process0 = out["config"]["process"][0]
    process0.pop("sim", None)
    return out


def _reject_cli_argv() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(
            "simulate_turbo_prior: no CLI arguments; "
            "edit simulate_turbo_prior.yaml (knobs live in the recipe)."
        )


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
            train_dtype = get_torch_dtype(getattr(self.train_config, "dtype", None))
            if not isinstance(train_dtype, torch.dtype):
                train_dtype = latents.dtype
            centers = centers.to(device=latents.device, dtype=train_dtype)
            t = centers[force_slot].expand(int(batch_size)).clone()
        else:
            t = _orig(self, batch_size, latents, step_num, content_or_style)
        expected_t = get_torch_dtype(getattr(self.train_config, "dtype", None))
        if isinstance(expected_t, torch.dtype) and t.dtype != expected_t:
            raise RuntimeError(
                f"Acceptance fail: sampled t dtype {t.dtype} != train.dtype {expected_t}"
            )
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
    """Append + print CUDA alloc/peak/reserved and DiT/TE/Turbo/VAE devices."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    total = _cuda_total_gb()
    frac = (peak / total) if total > 0 else 0.0
    gate = "FAIL" if total > 0 and frac >= PEAK_VRAM_FRAC_MAX else "ok"
    main_s = te_s = samp_s = vae_s = "?"
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
        vae = getattr(model, "vae", None)
        vd = _weight_device(vae) if vae is not None else None
        vae_s = str(vd) if vd is not None else "None"
    evt = (label, alloc, reserved, peak, total, main_s, te_s, samp_s, vae_s)
    _TE_CACHE_VRAM_EVENTS.append(evt)
    _log(
        f"[te-cache-vram] {label}: alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB "
        f"peak={peak:.2f}/{total:.2f}GiB ({frac:.1%} {gate}) "
        f"main={main_s} te={te_s} samp={samp_s} vae={vae_s}"
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
    for label, alloc, reserved, peak, total, main_s, te_s, samp_s, vae_s in _TE_CACHE_VRAM_EVENTS:
        dpeak = peak - prev_peak
        mark = " <<" if dpeak > 0.05 else ""
        frac = (peak / total) if total > 0 else 0.0
        gate = "FAIL" if total > 0 and frac >= PEAK_VRAM_FRAC_MAX else "ok"
        _log(
            f"  {label}: alloc={alloc:.2f} reserved={reserved:.2f} peak={peak:.2f} "
            f"(Δpeak={dpeak:+.2f}, {frac:.1%} {gate}) "
            f"main={main_s} te={te_s} samp={samp_s} vae={vae_s}{mark}"
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
    from toolkit.models.base_model import BaseModel
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
    _orig_set_preset = BaseModel.set_device_state_preset
    _orig_gen_single = ZImageDiffSynthModel.generate_single_image
    _orig_encode = ZImageDiffSynthModel.encode_prompt
    _encode_call_n = {"n": 0}
    _pipeline_before_logged = {"n": 0}

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
        quantize = bool(getattr(getattr(self, "model_config", None), "quantize", True))
        if not quantize:
            for name in ("network", "_sampling_network"):
                net = getattr(self, name, None)
                if net is not None and getattr(net, "can_merge_in", False):
                    raise RuntimeError(
                        "Acceptance fail: unquantized generate_images still has "
                        f"{name}.can_merge_in=True"
                    )
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

    def _set_preset(self, device_state_preset):
        out = _orig_set_preset(self, device_state_preset)
        if device_state_preset == "generate":
            _log_te_cache_vram("generate_preset:after", self)
        return out

    def _gen_single(self, pipeline, gen_config, conditional_embeds, unconditional_embeds, generator, extra):
        if _pipeline_before_logged["n"] == 0:
            _log_te_cache_vram("pipeline:before", self)
            _pipeline_before_logged["n"] = 1
        return _orig_gen_single(
            self,
            pipeline,
            gen_config,
            conditional_embeds,
            unconditional_embeds,
            generator,
            extra,
        )

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
    BaseModel.set_device_state_preset = _set_preset  # type: ignore[method-assign]
    ZImageDiffSynthModel.generate_single_image = _gen_single  # type: ignore[method-assign]
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
                    msg = (
                        f"mid-step CUDA alloc "
                        f"{alloc / (1024**3):.2f} GiB ≥ {PEAK_VRAM_FRAC_MAX:.0%} of "
                        f"{total / (1024**3):.2f} GiB"
                    )
                    if _QUANTIZE_HARD_PEAK:
                        raise RuntimeError(
                            f"Acceptance fail: {msg} (abort before TDR)"
                        )
                    _log(f"[vram] WARNING (quantize=false): {msg}")
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
        nc = getattr(self, "network_config", None)
        train = getattr(self, "train_config", None)
        raw = getattr(nc, "dtype", None) if nc is not None else None
        if raw is None and train is not None:
            raw = getattr(train, "dtype", None)
        expected = get_torch_dtype(raw)
        if isinstance(expected, torch.dtype):
            for p in network.parameters():
                if p.requires_grad and p.dtype != expected:
                    raise RuntimeError(
                        f"Acceptance fail: LoRA param dtype {p.dtype} != "
                        f"network.dtype {expected}"
                    )
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
    quantize: bool = True,
) -> None:
    """Hard GREEN asserts on DiT residency + peak VRAM fraction.

    When ``quantize`` is false, peak-VRAM ≥ max is a warning only (bf16 DiT
    legitimately uses more headroom); residency asserts still hard-fail.
    """
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
        msg = (
            f"CUDA peak {peak_gb:.2f} GiB ≥ "
            f"{PEAK_VRAM_FRAC_MAX:.0%} of {total_gb:.2f} GiB"
            + (
                f" (sticky load peak was {_PEAK_AFTER_LOAD_GB:.2f} GiB)"
                if _PEAK_AFTER_LOAD_GB is not None
                else ""
            )
        )
        if not quantize:
            _log(f"[vram] WARNING (quantize=false): {msg}")
        else:
            raise RuntimeError(f"Acceptance fail: {msg}")
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


def _prepare_job_config(
    recipe: dict,
    *,
    work_root: Path,
    dataset_dir: Path,
    profile: bool,
) -> dict:
    """Copy recipe, rewrite paths, apply profile overrides, strip ``sim``."""
    config = copy.deepcopy(recipe)
    process0 = config["config"]["process"][0]
    train = process0["train"]
    train_on_turbo = bool(train.get("turbo_teacher_weight", False))
    mode_tag = "turbo" if train_on_turbo else "base"
    batch_size = int(train.get("batch_size", 1) or 1)
    train_name = f"zimage_diffsynth_sim_turbo_prior_{mode_tag}_b{batch_size}"
    config["config"]["name"] = train_name

    output_root = work_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    process0["log_dir"] = str(output_root / "TensorBoard")
    process0["training_folder"] = str(output_root)
    process0["sqlite_db_path"] = str(work_root / "aitk_db.db")
    process0["datasets"][0]["folder_path"] = str(dataset_dir)

    # Env overrides for model paths (optional; yaml paths used when unset).
    model = process0["model"]
    env_model = os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip()
    env_samp = os.environ.get("ZIMAGE_DIFFSYNTH_SAMPLING_PATH", "").strip()
    if env_model:
        model["name_or_path"] = env_model
    if env_samp:
        model["sampling_name_or_path"] = env_samp
    if "ZIMAGE_DIFFSYNTH_TE_PATH" in os.environ:
        te = os.environ.get("ZIMAGE_DIFFSYNTH_TE_PATH", "").strip()
        if te:
            model["te_name_or_path"] = te
        else:
            model.pop("te_name_or_path", None)

    n_steps = int(train.get("steps", 1) or 1)
    if profile:
        # Profile needs enough steps for warmup + measure; disable sampling.
        if n_steps < PROFILE_TOTAL_STEPS:
            train["steps"] = PROFILE_TOTAL_STEPS
            n_steps = PROFILE_TOTAL_STEPS
        train["disable_sampling"] = True
        train["skip_first_sample"] = True
        if bool(train.get("unload_text_encoder")) is False:
            # Profile must keep TE unloaded (VRAM); do not remount for generate.
            train["unload_text_encoder"] = True
            _log("[sim] profile: forcing unload_text_encoder=true")
        process0["sample"]["sample_every"] = 10_000
        process0.setdefault("logging", {})["debug"] = False
        process0["save"]["save_every"] = max(n_steps, 1)

    return _strip_sim(config)


def _train_lora(
    work_root: Path,
    dataset_dir: Path,
    recipe: dict,
    *,
    profile: bool = False,
) -> Path:
    global _PROFILE_ENABLED, FORCE_COVERAGE_STEPS, _QUANTIZE_HARD_PEAK
    config = _prepare_job_config(
        recipe, work_root=work_root, dataset_dir=dataset_dir, profile=profile
    )
    process0 = config["config"]["process"][0]
    train = process0["train"]
    model = process0["model"]
    train_on_turbo = bool(train.get("turbo_teacher_weight", False))
    quantize = bool(model.get("quantize", True))
    _QUANTIZE_HARD_PEAK = bool(quantize)
    n_steps = int(train.get("steps", 1) or 1)
    train_name = config["config"]["name"]
    output_root = Path(process0["training_folder"])

    model_path = model.get("name_or_path") or ""
    sampling_path = model.get("sampling_name_or_path") or ""
    if not model_path or not os.path.isdir(str(model_path)):
        raise RuntimeError(f"Model path missing: {model_path!r}")
    if not sampling_path or not os.path.isdir(str(sampling_path)):
        raise RuntimeError(
            "Sampling (Turbo) path missing; required for _sampling_transformer PNGs."
        )

    force_coverage = min(FORCE_COVERAGE_STEPS, max(0, n_steps - 1))
    _saved_force = FORCE_COVERAGE_STEPS
    if profile or n_steps < FORCE_COVERAGE_STEPS:
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

    mk = model.get("model_kwargs") or {}
    _log(
        f"[PHASE TRAIN] run_job: start "
        f"(turbo_teacher_weight={train_on_turbo} "
        f"profile={bool(profile)} quantize={quantize} "
        f"steps={n_steps} rank={process0['network'].get('linear')} "
        f"res={process0['datasets'][0].get('resolution')} "
        f"dtype={train.get('dtype')} loader={mk.get('loader')} "
        f"refiners_off={mk.get('disable_noise_refiner')} "
        f"unload_te={train.get('unload_text_encoder')} "
        f"skip_first_sample={train.get('skip_first_sample')})"
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
    # Short / profile runs may miss full 8-slot coverage; keep VRAM/residency.
    if profile or n_steps < FORCE_COVERAGE_STEPS:
        _log(
            "[sim] skipping full t-slot acceptance "
            f"(force_coverage={force_coverage}, steps={n_steps})"
        )
    else:
        _assert_t_acceptance(slot_counts, frac_lt_300)
    _assert_vram_acceptance(
        peak_alloc,
        device_total,
        train_on_turbo=train_on_turbo,
        quantize=quantize,
    )

    save_dir = output_root / train_name
    candidates = list(save_dir.glob("*.safetensors"))
    if not candidates:
        raise RuntimeError(f"No LoRA checkpoint found in {save_dir}")
    lora_path = max(candidates, key=lambda p: p.stat().st_mtime)
    if n_steps >= 2:
        _assert_lora_delta(init_lora_path, lora_path)
    else:
        _log(
            f"[lora-delta] skip (steps={n_steps} < 2); checkpoint={lora_path}"
        )
    return lora_path


def _assert_pass_artifacts(
    work_root: Path,
    batch_size: int,
    *,
    train_on_turbo: bool,
    profile: bool = False,
    skip_first_sample: bool = True,
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
            "sample PNGs skipped (sim.profile disables sampling)"
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
        if skip_first_sample:
            _log(
                f"   [{mode_tag}] LoRA OK: {lora_path}; "
                "no sample PNGs (skip_first_sample=true)"
            )
            return
        raise RuntimeError(f"No sample PNGs found under {samples_dir}")
    _log(
        f"   [{mode_tag}] LoRA OK: {lora_path}; "
        f"PNG(s): {[str(p) for p in train_samples]}"
    )


def _run_single_pass(
    *,
    work_root: Path,
    dataset_dir: Path,
    recipe: dict,
    profile: bool = False,
) -> None:
    process0 = recipe["config"]["process"][0]
    train = process0["train"]
    train_on_turbo = bool(train.get("turbo_teacher_weight", False))
    batch_size = int(train.get("batch_size", 1) or 1)
    skip_first = bool(train.get("skip_first_sample", True))
    mode = "true" if train_on_turbo else "false"
    _log(
        f"[pass] turbo_teacher_weight={mode} work={work_root} "
        f"profile={profile}"
    )
    work_root.mkdir(parents=True, exist_ok=True)
    _train_lora(
        work_root,
        dataset_dir,
        recipe,
        profile=profile,
    )
    _assert_pass_artifacts(
        work_root,
        batch_size,
        train_on_turbo=train_on_turbo,
        profile=profile,
        skip_first_sample=skip_first,
    )


def main() -> None:
    _reject_cli_argv()

    # Child worker: one GPU pass then exit (CUDA isolation via subprocess).
    if os.environ.get("SIM_TURBO_PRIOR_CHILD", "").strip() in ("1", "true", "yes"):
        faulthandler.enable()
        work_root = Path(os.environ["SIM_TURBO_PRIOR_WORK"])
        dataset_dir = Path(os.environ["SIM_TURBO_PRIOR_DATASET"])
        recipe = _load_recipe(_default_yaml_path())
        process0 = recipe["config"]["process"][0]
        sim = _parse_sim(process0)
        profile = bool(sim["profile"])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; GPU required for this sim.")
        _run_single_pass(
            work_root=work_root,
            dataset_dir=dataset_dir,
            recipe=recipe,
            profile=profile,
        )
        return

    recipe = _load_recipe(_default_yaml_path())
    process0 = recipe["config"]["process"][0]
    sim = _parse_sim(process0)
    profile = bool(sim["profile"])
    train_on_turbo = bool(process0["train"].get("turbo_teacher_weight", False))
    mode = "true" if train_on_turbo else "false"
    _log(
        "Z-Image DiffSynth simulate_turbo_prior "
        f"(yaml={_default_yaml_path().name}; "
        f"timestep_type=turbo_prior; turbo={mode}; "
        f"profile={profile}; quantize={process0['model'].get('quantize')}) ..."
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; GPU required for this sim.")

    prompt = os.environ.get("ZIMAGE_TEST_TRAIN_PROMPT", "dog")
    seeds = [42 + i for i in range(NUM_SOURCE_IMAGES)]
    image_cache = TEST_TRAIN_IMAGE_CACHE

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

    _log(
        "[sim] use_diffsynth_prompt_encoding omitted → true "
        "(turbo_prior DiffSynth encoding locked on)"
    )
    te_path = process0["model"].get("te_name_or_path")
    if te_path:
        _log(f"[sim] te_name_or_path={te_path!r}")
    work_root = base_work / ("turbo" if train_on_turbo else "base")
    work_root.mkdir(parents=True, exist_ok=True)
    _log(
        f"2) Single pass: turbo_teacher_weight={mode} "
        f"(work={work_root}; fresh subprocess for CUDA isolation) ..."
    )
    child_env = os.environ.copy()
    child_env["SIM_TURBO_PRIOR_CHILD"] = "1"
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
