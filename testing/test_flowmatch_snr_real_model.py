"""
Real zimage_diffsynth training path: compare MSE loss (pre-SNR) vs optimizer loss (post-SNR).

Forces scheduler slot indices on the live BatchProcessor -> TimestepSampler path, runs one
optimizer step with the real DiT + LoRA stack, and records:
  - per-sample MSE mean entering apply_snr_weight
  - per-sample loss after min_snr_gamma
  - scalar from calculate_loss (before extra train_single_accumulation scaling)
  - scalar immediately before accelerator.backward

Run (repo root, venv, CUDA + local Z-Image weights + temp/test_train images):
  python -m pytest testing/test_flowmatch_snr_real_model.py -v -s

Env: ZIMAGE_DIFFSYNTH_MODEL_PATH, ZIMAGE_DIFFSYNTH_SAMPLING_PATH (same as test_smoke).
"""

from __future__ import annotations

import importlib
import json
import os
import gc
import subprocess
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (  # noqa: E402
    DEFAULT_ZIMAGE_MODEL_PATH,
    DEFAULT_ZIMAGE_SAMPLING_PATH,
)
from toolkit.job import run_job  # noqa: E402
from toolkit.timestep_sampler import TimestepSampler, TimestepSamplerResult  # noqa: E402

SNR_REAL_MODEL_PROBES: list[tuple[int, float]] = [(1, 1.0), (999, 1.0)]
_BASELINE_SUBPROCESS_ENV = "ZIMAGE_FLOWMATCH_BASELINE_SUBPROCESS"
_BASELINE_BATCH_SIZE_ENV = "ZIMAGE_FLOWMATCH_BASELINE_BATCH_SIZE"
_BASELINE_WORK_ROOT_ENV = "ZIMAGE_FLOWMATCH_BASELINE_WORK_ROOT"
_BASELINE_OUTPUT_JSON_ENV = "ZIMAGE_FLOWMATCH_BASELINE_OUTPUT_JSON"


def _resolve_model_paths() -> tuple[str, str | None]:
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
    return model_path, sampling_path


def _user_flowmatch_job_config(
    work_root: Path,
    dataset_dir: Path,
    model_path: str,
    sampling_path: str | None,
    *,
    min_snr_gamma: float,
    steps: int = 1,
    batch_size: int = 1,
    debug: bool = False,
) -> dict:
    output_root = work_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    proc: dict = {
        "type": "z_image_diffsynth_trainer",
        "training_folder": str(output_root),
        "sqlite_db_path": str(work_root / "aitk_db.db"),
        "device": "cuda",
        "trigger_word": None,
        "performance_log_every": 0,
        "network": {
            "rank_dropout": 0.1,
            "type": "lora",
            "linear": 128,
            "linear_alpha": 128,
            "conv": 0,
            "conv_alpha": 0,
        },
        "save": {
            "dtype": "bf16",
            "save_every": 10_000,
            "max_step_saves_to_keep": 1,
            "save_format": "safetensors",
            "push_to_hub": False,
        },
        "train": {
            "batch_size": batch_size,
            "gradient_accumulation": 1,
            "steps": steps,
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": True,
            "noise_scheduler": "flowmatch",
            "prediction_type": "flowmatch",
            "timestep_type": "shift",
            "timestep_weighting": "none",
            "content_or_style": "balanced",
            "optimizer": "adafactor",
            "loss_type": "mse",
            "lr": 0.0001,
            "min_snr_gamma": min_snr_gamma,
            "min_denoising_steps": 1,
            "max_denoising_steps": 1000,
            "optimizer_params": {
                "beta2": 0.9,
                "weight_decay": 0.14,
                "scale_parameter": False,
                "relative_step": False,
                "beta1": 0,
            },
            "unload_text_encoder": True,
            "cache_text_embeddings": True,
            "skip_first_sample": True,
            "disable_sampling": True,
            "dtype": "bf16",
        },
        "logging": {"log_every": 0, "use_ui_logger": False, "debug": debug},
        "model": {
            "name_or_path": model_path,
            "quantize": True,
            "qtype": "qfloat8",
            "quantize_te": True,
            "qtype_te": "qfloat8",
            "arch": "zimage_diffsynth",
            "model_kwargs": {
                "use_diffsynth_training_loop": False,
                "use_dynamic_shifting": True,
            },
        },
        "datasets": [
            {
                "folder_path": str(dataset_dir),
                "cache_latents_to_disk": True,
                "resolution": [1024],
                "num_repeats": 1,
            }
        ],
    }
    if sampling_path:
        proc["model"]["sampling_name_or_path"] = sampling_path
    return {"job": "extension", "config": {"name": "test_flowmatch_snr_real", "process": [proc]}}


@dataclass
class LossProbeRecord:
    forced_slot: int
    timestep_value: float
    slot_index: int | None
    mse_per_sample: float
    post_snr_per_sample: float
    snr_weight: float
    calculate_loss_scalar: float
    pre_backward_scalar: float
    prediction_type: str
    min_snr_gamma: float
    hypo_gamma1_post_snr: float
    hypo_gamma5_post_snr: float


@dataclass
class LossProbeState:
    force_slots_by_step: dict[int, int] = field(default_factory=dict)
    current_forced_slot: int | None = None
    records: list[LossProbeRecord] = field(default_factory=list)
    calculate_loss_scalar: float | None = None
    pre_backward_scalar: float | None = None
    last_timestep_value: float | None = None
    last_slot_index: int | None = None
    stage_peak_alloc_mb: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    stage_peak_reserved_mb: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class ProbeRunResult:
    by_slot: dict[int, LossProbeRecord]
    peak_allocated_mb: float
    peak_reserved_mb: float
    stage_peak_allocated_mb: dict[str, float]
    stage_peak_reserved_mb: dict[str, float]


def _install_probes(monkeypatch: pytest.MonkeyPatch, state: LossProbeState) -> None:
    sdtrainer_module = importlib.import_module("extensions_built_in.sd_trainer.SDTrainer")
    base_process_module = importlib.import_module("jobs.process.BaseSDTrainProcess")

    orig_sample = TimestepSampler.sample
    orig_apply_snr = sdtrainer_module.apply_snr_weight
    orig_calculate_loss = sdtrainer_module.SDTrainer.calculate_loss
    orig_train_single = sdtrainer_module.SDTrainer.train_single_accumulation
    orig_predict_noise = sdtrainer_module.SDTrainer.predict_noise
    orig_process_batch = base_process_module.BaseSDTrainProcess.process_general_training_batch

    def _record_cuda_peak(stage: str):
        if not torch.cuda.is_available():
            return
        state.stage_peak_alloc_mb[stage].append(
            torch.cuda.max_memory_allocated() / 2**20
        )
        state.stage_peak_reserved_mb[stage].append(
            torch.cuda.max_memory_reserved() / 2**20
        )

    def _forced_sample(
        self,
        batch_size,
        latents,
        content_or_style,
        min_noise_steps,
        max_noise_steps,
        num_train_timesteps,
        device,
        step_num,
    ):
        if step_num not in state.force_slots_by_step:
            state.current_forced_slot = None
            return orig_sample(
                self,
                batch_size,
                latents,
                content_or_style,
                min_noise_steps,
                max_noise_steps,
                num_train_timesteps,
                device,
                step_num,
            )
        slot = int(state.force_slots_by_step[step_num])
        state.current_forced_slot = slot
        indices = torch.full((batch_size,), slot, device=device, dtype=torch.long)
        timesteps = self.noise_scheduler.timesteps[indices.long()]
        return TimestepSamplerResult(timesteps=timesteps, timestep_indices=indices)

    def _wrapped_process_batch(self, batch):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        out = orig_process_batch(self, batch)
        _record_cuda_peak("prepare_batch")
        try:
            _, _, timesteps, _, _ = out
            ts = timesteps.detach().float().view(-1)
            state.last_timestep_value = float(ts[0].item())
            sched = self.sd.noise_scheduler.timesteps.detach().float().cpu()
            state.last_slot_index = int((sched - ts[0].cpu()).abs().argmin().item())
        except Exception:
            state.last_timestep_value = None
            state.last_slot_index = None
        return out

    def _wrapped_predict_noise(self, *args, **kwargs):
        is_primary = bool(kwargs.get("is_primary_pred", False))
        if is_primary and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        out = orig_predict_noise(self, *args, **kwargs)
        if is_primary:
            _record_cuda_peak("forward")
        return out

    def _wrapped_apply_snr_weight(
        loss,
        timesteps,
        noise_scheduler,
        gamma,
        fixed=False,
        prediction_type="epsilon",
    ):
        ts = torch.as_tensor(timesteps).detach().float().view(-1)
        out = orig_apply_snr(
            loss,
            timesteps,
            noise_scheduler,
            gamma,
            fixed=fixed,
            prediction_type=prediction_type,
        )
        with torch.no_grad():
            base = loss.detach().float().view(-1)
            post = out.detach().float().view(-1)
            mse_val = float(base.mean().item())
            post_val = float(post.mean().item())
            w_val = float((post / base.clamp(min=1e-30)).mean().item())
            h1 = float(
                orig_apply_snr(
                    base.clone(),
                    ts,
                    noise_scheduler,
                    1.0,
                    fixed=fixed,
                    prediction_type=prediction_type,
                ).mean().item()
            )
            h5 = float(
                orig_apply_snr(
                    base.clone(),
                    ts,
                    noise_scheduler,
                    5.0,
                    fixed=fixed,
                    prediction_type=prediction_type,
                ).mean().item()
            )
        state.records.append(
            LossProbeRecord(
                forced_slot=int(state.current_forced_slot if state.current_forced_slot is not None else -1),
                timestep_value=float(ts[0].item()),
                slot_index=state.last_slot_index,
                mse_per_sample=mse_val,
                post_snr_per_sample=post_val,
                snr_weight=w_val,
                calculate_loss_scalar=float("nan"),
                pre_backward_scalar=float("nan"),
                prediction_type=str(prediction_type),
                min_snr_gamma=float(gamma),
                hypo_gamma1_post_snr=h1,
                hypo_gamma5_post_snr=h5,
            )
        )
        return out

    def _wrapped_calculate_loss(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        out = orig_calculate_loss(self, *args, **kwargs)
        _record_cuda_peak("loss")
        state.calculate_loss_scalar = float(out.detach().float().item())
        if state.records:
            state.records[-1].calculate_loss_scalar = state.calculate_loss_scalar
        return out

    def _wrapped_train_single(self, batch, microbatch_scale: float = 1.0):
        importlib.import_module("accelerate")

        orig_backward = self.accelerator.backward
        captured: dict[str, float] = {}

        def _capture_backward(loss_tensor, *bargs, **bkwargs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            captured["pre_backward"] = float(loss_tensor.detach().float().item())
            out = orig_backward(loss_tensor, *bargs, **bkwargs)
            _record_cuda_peak("backward")
            return out

        self.accelerator.backward = _capture_backward
        try:
            result = orig_train_single(self, batch, microbatch_scale=microbatch_scale)
        finally:
            self.accelerator.backward = orig_backward
        if "pre_backward" in captured:
            state.pre_backward_scalar = captured["pre_backward"]
            if state.records:
                state.records[-1].pre_backward_scalar = captured["pre_backward"]
        return result

    monkeypatch.setattr(TimestepSampler, "sample", _forced_sample)
    monkeypatch.setattr(sdtrainer_module, "apply_snr_weight", _wrapped_apply_snr_weight)
    monkeypatch.setattr(sdtrainer_module.SDTrainer, "calculate_loss", _wrapped_calculate_loss)
    monkeypatch.setattr(sdtrainer_module.SDTrainer, "train_single_accumulation", _wrapped_train_single)
    monkeypatch.setattr(sdtrainer_module.SDTrainer, "predict_noise", _wrapped_predict_noise)
    monkeypatch.setattr(
        base_process_module.BaseSDTrainProcess,
        "process_general_training_batch",
        _wrapped_process_batch,
    )


def _run_probes_in_one_job(
    work_root: Path,
    probes: list[tuple[int, float]],
    *,
    batch_size: int = 1,
    enable_debug: bool = False,
) -> ProbeRunResult:
    model_path, sampling_path = _resolve_model_paths()
    dataset_dir = REPO_ROOT / "temp" / "test_train"
    gammas = {g for _, g in probes}
    if len(gammas) != 1:
        raise ValueError("all probes must share the same min_snr_gamma for a single job run")
    min_snr_gamma = probes[0][1]
    force_slots_by_step = {i: slot for i, (slot, _) in enumerate(probes)}
    config = _user_flowmatch_job_config(
        work_root,
        dataset_dir,
        model_path,
        sampling_path,
        min_snr_gamma=min_snr_gamma,
        steps=len(probes),
        batch_size=batch_size,
        debug=enable_debug,
    )
    state = LossProbeState(force_slots_by_step=force_slots_by_step)
    monkeypatch = pytest.MonkeyPatch()
    try:
        _install_probes(monkeypatch, state)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        run_job(config)
    finally:
        monkeypatch.undo()

    assert len(state.records) == len(probes), (
        f"expected {len(probes)} apply_snr_weight hits, got {len(state.records)}"
    )
    by_slot = {rec.forced_slot: rec for rec in state.records}
    for slot, _ in probes:
        assert slot in by_slot, f"apply_snr_weight was not reached for slot {slot}"
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if torch.cuda.is_available():
        peak_allocated_mb = torch.cuda.max_memory_allocated() / 2**20
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 2**20
    stage_peak_allocated_mb = {
        key: max(values) for key, values in state.stage_peak_alloc_mb.items() if values
    }
    stage_peak_reserved_mb = {
        key: max(values) for key, values in state.stage_peak_reserved_mb.items() if values
    }
    return ProbeRunResult(
        by_slot=by_slot,
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,
        stage_peak_allocated_mb=stage_peak_allocated_mb,
        stage_peak_reserved_mb=stage_peak_reserved_mb,
    )


def _drop_accelerate_global_singleton() -> None:
    try:
        import toolkit.accelerator as acc
    except Exception:
        return
    acc.global_accelerator = None


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _serialize_probe_result(result: ProbeRunResult) -> dict:
    return {
        "peak_allocated_mb": float(result.peak_allocated_mb),
        "peak_reserved_mb": float(result.peak_reserved_mb),
        "stage_peak_allocated_mb": {
            key: float(value) for key, value in result.stage_peak_allocated_mb.items()
        },
        "stage_peak_reserved_mb": {
            key: float(value) for key, value in result.stage_peak_reserved_mb.items()
        },
    }


def _run_batch_memory_baseline_subprocess(
    work_root: Path,
    *,
    batch_size: int,
) -> dict:
    output_json = work_root / f"batch_probe_b{batch_size}.json"
    env = os.environ.copy()
    env[_BASELINE_SUBPROCESS_ENV] = "1"
    env[_BASELINE_BATCH_SIZE_ENV] = str(batch_size)
    env[_BASELINE_WORK_ROOT_ENV] = str(work_root / f"worker_b{batch_size}")
    env[_BASELINE_OUTPUT_JSON_ENV] = str(output_json)

    cmd = [sys.executable, "-m", "testing.test_flowmatch_snr_real_model"]
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    if not output_json.is_file():
        raise RuntimeError(f"baseline subprocess did not write {output_json}")
    return json.loads(output_json.read_text(encoding="utf-8"))


def _baseline_subprocess_worker_main() -> None:
    batch_size = int(os.environ[_BASELINE_BATCH_SIZE_ENV])
    work_root = Path(os.environ[_BASELINE_WORK_ROOT_ENV])
    output_json = Path(os.environ[_BASELINE_OUTPUT_JSON_ENV])
    work_root.mkdir(parents=True, exist_ok=True)

    probes = [(999, 1.0)]
    try:
        result = _run_probes_in_one_job(
            work_root,
            probes,
            batch_size=batch_size,
            enable_debug=True,
        )
        payload = _serialize_probe_result(result)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        _drop_accelerate_global_singleton()
        _release_cuda()


def _print_probe_record(slot: int, rec: LossProbeRecord) -> None:
    print(
        f"\n[slot {slot}] ts={rec.timestep_value:.4g} "
        f"mse={rec.mse_per_sample:.6g} post_snr={rec.post_snr_per_sample:.6g} "
        f"w={rec.snr_weight:.6g} calc_loss={rec.calculate_loss_scalar:.6g} "
        f"pre_bwd={rec.pre_backward_scalar:.6g} "
        f"hypo_g1={rec.hypo_gamma1_post_snr:.6g} hypo_g5={rec.hypo_gamma5_post_snr:.6g}",
        flush=True,
    )


def test_high_noise_slot(snr_probe_records: ProbeRunResult):
    """
    Slot 1 -> scheduler value ~999 (max-noise end): MSE != post-SNR, but gamma1 == gamma5 post-SNR.
    """
    rec_hi = snr_probe_records.by_slot[1]
    _print_probe_record(1, rec_hi)

    assert rec_hi.timestep_value > 900.0, "slot 1 must map to high-noise scheduler value (~999)"
    assert rec_hi.mse_per_sample != pytest.approx(rec_hi.post_snr_per_sample, rel=1e-3)
    assert rec_hi.calculate_loss_scalar == pytest.approx(rec_hi.pre_backward_scalar, rel=1e-6)
    assert rec_hi.hypo_gamma1_post_snr == pytest.approx(rec_hi.hypo_gamma5_post_snr, rel=0, abs=1e-7)
    assert rec_hi.snr_weight < 1e-5


def test_low_noise_slot(snr_probe_records: ProbeRunResult):
    """
    Slot 999 -> scheduler value ~3 (min-noise end): MSE != post-SNR, gamma1 != gamma5 post-SNR.
    """
    rec_lo = snr_probe_records.by_slot[999]
    _print_probe_record(999, rec_lo)

    assert rec_lo.timestep_value < 10.0, "slot 999 must map to low-noise scheduler value (~3)"
    assert rec_lo.mse_per_sample != pytest.approx(rec_lo.post_snr_per_sample, rel=1e-3)
    assert rec_lo.calculate_loss_scalar == pytest.approx(rec_lo.pre_backward_scalar, rel=1e-6)
    assert rec_lo.hypo_gamma5_post_snr > rec_lo.hypo_gamma1_post_snr * 4.0


def test_batch_size_memory_baseline_b1_vs_b2():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for memory baseline comparison")
    model_path, _ = _resolve_model_paths()
    if not model_path or not os.path.isdir(model_path):
        pytest.skip(
            "Z-Image model path is missing. Set ZIMAGE_DIFFSYNTH_MODEL_PATH or update defaults."
        )

    # Keep heavy subprocess artifacts on the repo drive (not pytest's system temp on C:)
    # to avoid WinError 112 on machines with a small system partition.
    work_root = (
        REPO_ROOT
        / "temp"
        / "flowmatch_batch_memory_baseline"
        / f"pytest_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    work_root.mkdir(parents=True, exist_ok=True)
    run_b1 = _run_batch_memory_baseline_subprocess(work_root, batch_size=1)
    _drop_accelerate_global_singleton()
    _release_cuda()
    run_b2 = _run_batch_memory_baseline_subprocess(work_root, batch_size=2)

    for stage in ("prepare_batch", "forward", "loss", "backward"):
        assert stage in run_b1["stage_peak_allocated_mb"], f"missing stage peak for B=1: {stage}"
        assert stage in run_b2["stage_peak_allocated_mb"], f"missing stage peak for B=2: {stage}"

    assert run_b1["peak_allocated_mb"] > 0.0
    assert run_b2["peak_allocated_mb"] > 0.0
    # Allow minor allocator noise; B=2 should not consume substantially less peak VRAM than B=1.
    assert run_b2["peak_allocated_mb"] >= run_b1["peak_allocated_mb"] * 0.90


if __name__ == "__main__":
    if os.environ.get(_BASELINE_SUBPROCESS_ENV, "").strip() == "1":
        _baseline_subprocess_worker_main()
