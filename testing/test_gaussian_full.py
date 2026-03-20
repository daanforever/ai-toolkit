"""
Full-stack test: real z_image_diffsynth_trainer + gaussian_bimodal timestep sampling.

Collects scheduler slot indices from the live training loop (same path as production:
BatchProcessor -> TimestepSampler -> gaussian bimodal on index grid) via a monkeypatch
on TimestepDistributionLogger.collect.

After training, writes TensorBoard-style scalars JSON to ``temp/test_gaussian_full.json``
(same structure as ``scripts/tensorboard_extract.py``) for
``testing/verify_flowmatch_bimodal_scalars.py`` (expects ``timestep_weights/min_timestep``).

Requires CUDA, local Z-Image weights, and dataset under repo ``temp/test_train/`` (same
convention as ``extensions_built_in/.../test_train.py``). Paths default to
``test_smoke.DEFAULT_*``; override with ZIMAGE_DIFFSYNTH_MODEL_PATH /
ZIMAGE_DIFFSYNTH_SAMPLING_PATH.

Run (repo root, venv):
  python -m pytest testing/test_gaussian_full.py -v

Analyze export (repo root, venv):
  python -m testing.verify_flowmatch_bimodal_scalars temp/test_gaussian_full.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.platform == "win32":
    _venv_python = REPO_ROOT / "venv" / "Scripts" / "python.exe"
else:
    _venv_python = REPO_ROOT / "venv" / "bin" / "python"
if _venv_python.is_file():
    _current = os.path.realpath(sys.executable)
    _venv_real = os.path.realpath(str(_venv_python))
    if _current != _venv_real:
        os.execv(_venv_real, [_venv_real] + sys.argv)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

import pytest
import torch

from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (  # noqa: E402
    DEFAULT_ZIMAGE_MODEL_PATH,
    DEFAULT_ZIMAGE_SAMPLING_PATH,
)
from extensions_built_in.sd_trainer.gaussian_timestep_weights import (  # noqa: E402
    evaluate_gaussian_timestep_bimodal,
    scheduler_timesteps_align_with_index_grid,
    timestep_values_to_slot_indices,
)
from toolkit.job import run_job  # noqa: E402
from toolkit.timestep_debug import TimestepDistributionLogger  # noqa: E402
from toolkit.timestep_sampler import allowed_slot_index_range  # noqa: E402

TB_JSON_REL = Path("temp") / "test_gaussian_full.json"
TB_TAGS = (
    "timestep_weights/min_timestep",
    "timestep_weights/max_timestep",
    "timestep_weights/mean_timestep",
    "timestep_weights/mean_weight",
)


def _empty_tb_payload() -> dict[str, list[dict]]:
    return {tag: [] for tag in TB_TAGS}


def _loss_weights_bimodal_like_sdtrainer(
    timesteps: torch.Tensor,
    train_config,
    noise_scheduler,
) -> torch.Tensor | None:
    """Match SDTrainer gaussian_bimodal branch (slot lookup vs raw values)."""
    if noise_scheduler is None:
        return None
    ntt = int(noise_scheduler.config.num_train_timesteps)
    schedule = noise_scheduler.timesteps
    aligned = scheduler_timesteps_align_with_index_grid(schedule, ntt)
    sched_on_dev = schedule.to(device=timesteps.device, dtype=torch.float32)
    lookup = (
        timestep_values_to_slot_indices(
            timesteps.detach().float(), sched_on_dev, ntt=ntt
        )
        if not aligned
        else timesteps.detach().float()
    )
    return evaluate_gaussian_timestep_bimodal(
        lookup,
        train_config.gaussian_mean,
        train_config.gaussian_std,
        train_config.gaussian_mean_2,
        train_config.gaussian_std_2,
        timesteps.device,
        torch.float32,
        ntt,
    )


def _append_tb_scalars(
    payload: dict[str, list[dict]],
    *,
    step_num: int,
    timesteps: torch.Tensor,
    weights: torch.Tensor | None,
) -> None:
    """One training micro-batch → one row per tag (TensorBoard scalar JSON shape)."""
    t = timesteps.detach().cpu().flatten().float()
    wt = time.time()
    payload["timestep_weights/min_timestep"].append(
        {"step": int(step_num), "value": float(t.min().item()), "wall_time": wt}
    )
    payload["timestep_weights/max_timestep"].append(
        {"step": int(step_num), "value": float(t.max().item()), "wall_time": wt}
    )
    payload["timestep_weights/mean_timestep"].append(
        {"step": int(step_num), "value": float(t.mean().item()), "wall_time": wt}
    )
    if weights is not None:
        w = weights.detach().cpu().flatten().float()
        payload["timestep_weights/mean_weight"].append(
            {"step": int(step_num), "value": float(w.mean().item()), "wall_time": wt}
        )
    else:
        payload["timestep_weights/mean_weight"].append(
            {"step": int(step_num), "value": 1.0, "wall_time": wt}
        )


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


def _gaussian_full_job_config(
    work_root: Path,
    dataset_dir: Path,
    model_path: str,
    sampling_path: str | None,
) -> dict:
    """User-provided training config; only paths and output locations are substituted."""
    output_root = work_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    proc: dict = {
        "type": "z_image_diffsynth_trainer",
        "training_folder": str(output_root),
        "sqlite_db_path": str(work_root / "aitk_db.db"),
        "device": "cuda",
        "trigger_word": None,
        "performance_log_every": 10,
        "network": {
            "rank_dropout": 0.01,
            "type": "lora",
            "linear": 32,
            "linear_alpha": 1,
            "conv": 0,
            "conv_alpha": 0,
            "lokr_full_rank": False,
            "lokr_factor": -1,
            "network_kwargs": {
                "ignore_if_contains": [],
                "lora_down_init_scale": 1,
                "init_lora_weights": "pissa",
            },
            "pretrained_lora_path": "",
        },
        "save": {
            "dtype": "bf16",
            "save_every": 50,
            "max_step_saves_to_keep": 2,
            "save_format": "safetensors",
            "push_to_hub": False,
        },
        "train": {
            "lr": 0.00005,
            "noise_offset": 0.1,
            "max_denoising_steps": 995,
            "min_denoising_steps": 5,
            "batch_size": 1,
            "bypass_guidance_embedding": False,
            "steps": 100,
            "gradient_accumulation": 1,
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": True,
            "noise_scheduler": "flowmatch",
            "optimizer": "adafactor",
            "timestep_type": "gaussian_bimodal",
            "content_or_style": "gaussian_bimodal",
            "gaussian_mean": 300,
            "gaussian_std": 0.2,
            "optimizer_params": {
                "emergency_brake": 1,
                "beta2": 0.99,
                "weight_decay": 0.001,
                "scale_parameter": False,
                "relative_step": True,
                "warmup_init": True,
                "warmup_steps": 100,
                "min_lr": 0,
                "rms_max_decay_rate": 0.99,
                "stochastic_accumulation": True,
                "stochastic_rounding": True,
                "factored": True,
                "beta1": 0.9,
                "saddle_point_window": 100,
                "saddle_point_threshold": 0.000001,
                "saddle_point_step": 0.5,
            },
            "unload_text_encoder": False,
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
            "min_snr_gamma": 5,
            "gaussian_mean_2": 800,
            "gaussian_std_2": 0.2,
        },
        "logging": {"log_every": 1, "use_ui_logger": True, "debug": True},
        "model": {
            "debug_zimage_load": False,
            "name_or_path": model_path,
            "quantize": True,
            "qtype": "qfloat8",
            "quantize_te": False,
            "qtype_te": "qfloat8",
            "arch": "zimage_diffsynth",
            "low_vram": False,
            "model_kwargs": {
                "use_diffsynth_training_loop": False,
                "disable_noise_refiner": False,
                "disable_context_refiner": False,
                "sampling_loader": "diffusers",
            },
            "layer_offloading": False,
            "layer_offloading_text_encoder_percent": 1,
            "layer_offloading_transformer_percent": 1,
        },
        "datasets": [
            {
                "folder_path": str(dataset_dir),
                "tone_correction": True,
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
            "sample_every": 50,
            "width": 1024,
            "height": 768,
            "samples": [{"prompt": "dog"}],
            "neg": "",
            "seed": 42,
            "walk_seed": True,
            "guidance_scale": 1,
            "sample_steps": 9,
            "num_frames": 1,
            "fps": 1,
        },
    }
    if sampling_path:
        proc["model"]["sampling_name_or_path"] = sampling_path

    return {
        "job": "extension",
        "config": {
            "name": "test_gaussian",
            "process": [proc],
        },
    }


def _expected_index_moments_and_bin_probs(
    *,
    ntt: int,
    min_noise_steps: int,
    max_noise_steps: int,
    gaussian_mean: float,
    gaussian_std: float,
    gaussian_mean_2: float,
    gaussian_std_2: float,
) -> tuple[float, float, torch.Tensor]:
    lo, hi = allowed_slot_index_range(ntt, min_noise_steps, max_noise_steps)
    all_indices = torch.arange(lo, hi + 1, dtype=torch.float32)
    weights = evaluate_gaussian_timestep_bimodal(
        all_indices,
        gaussian_mean,
        gaussian_std,
        gaussian_mean_2,
        gaussian_std_2,
        torch.device("cpu"),
        torch.float32,
        ntt,
    )
    probs = weights / weights.sum().clamp(min=1e-8)
    idx_long = all_indices.long()
    mean_idx = (all_indices * probs).sum().item()
    var_idx = (((all_indices - mean_idx) ** 2) * probs).sum().item()
    std_idx = var_idx**0.5

    # Three bins: low / mid / high — first mode ~300, second ~800 on a 0..ntt-1 style axis.
    m1 = idx_long <= 500
    m2 = (idx_long > 500) & (idx_long <= 700)
    m3 = idx_long > 700
    bin_probs = torch.stack([probs[m1].sum(), probs[m2].sum(), probs[m3].sum()])
    return mean_idx, std_idx, bin_probs


def _chi2_three_bins(captured: list[int], bin_probs: torch.Tensor) -> float:
    n = len(captured)
    idx = torch.tensor(captured, dtype=torch.long)
    o1 = int((idx <= 500).sum().item())
    o2 = int(((idx > 500) & (idx <= 700)).sum().item())
    o3 = int((idx > 700).sum().item())
    exp = (bin_probs * n).clamp(min=1e-6)
    obs = torch.tensor([o1, o2, o3], dtype=torch.float32)
    return float((((obs - exp) ** 2) / exp).sum().item())


def test_gaussian_bimodal_timesteps_match_index_distribution(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model_path, sampling_path = _resolve_model_paths()
    if not model_path or not os.path.isdir(model_path):
        pytest.skip(
            "Z-Image model path missing or invalid "
            "(set ZIMAGE_DIFFSYNTH_MODEL_PATH or DEFAULT_ZIMAGE_MODEL_PATH)"
        )

    dataset_dir = REPO_ROOT / "temp" / "test_train"
    if not dataset_dir.is_dir():
        pytest.skip(f"Dataset folder missing: {dataset_dir}")
    pngs = list(dataset_dir.glob("*.png"))
    if not pngs:
        pytest.skip(f"No images in {dataset_dir}")

    work_root = Path(tmp_path)
    config = _gaussian_full_job_config(work_root, dataset_dir, model_path, sampling_path)

    train_cfg = config["config"]["process"][0]["train"]
    ntt = 1000
    min_ns = int(train_cfg["min_denoising_steps"])
    max_ns = int(train_cfg["max_denoising_steps"])
    mu1 = float(train_cfg["gaussian_mean"])
    s1 = float(train_cfg["gaussian_std"])
    mu2 = float(train_cfg["gaussian_mean_2"])
    s2 = float(train_cfg["gaussian_std_2"])

    mean_exp, std_exp, bin_probs = _expected_index_moments_and_bin_probs(
        ntt=ntt,
        min_noise_steps=min_ns,
        max_noise_steps=max_ns,
        gaussian_mean=mu1,
        gaussian_std=s1,
        gaussian_mean_2=mu2,
        gaussian_std_2=s2,
    )
    lo, hi = allowed_slot_index_range(ntt, min_ns, max_ns)

    captured: list[int] = []
    tb_payload = _empty_tb_payload()
    orig_collect = TimestepDistributionLogger.collect

    def wrapped_collect(self, timestep_indices, timesteps, content_or_style, step_num, timestep_sampler):
        if timestep_indices is not None:
            captured.extend(timestep_indices.detach().cpu().tolist())
        ns = getattr(timestep_sampler, "noise_scheduler", None)
        w_tensor = None
        if self.train_config.timestep_type == "gaussian_bimodal" and ns is not None:
            w_tensor = _loss_weights_bimodal_like_sdtrainer(
                timesteps, self.train_config, ns
            )
        _append_tb_scalars(
            tb_payload,
            step_num=int(step_num),
            timesteps=timesteps,
            weights=w_tensor,
        )
        return orig_collect(
            self,
            timestep_indices,
            timesteps,
            content_or_style,
            step_num,
            timestep_sampler,
        )

    monkeypatch.setattr(TimestepDistributionLogger, "collect", wrapped_collect)

    torch.manual_seed(42)
    run_job(config)

    tb_out = REPO_ROOT / TB_JSON_REL
    tb_out.parent.mkdir(parents=True, exist_ok=True)
    with open(tb_out, "w", encoding="utf-8") as f:
        json.dump(tb_payload, f, indent=2, ensure_ascii=False)

    assert len(captured) >= 50, f"expected many timestep samples, got {len(captured)}"

    t = torch.tensor(captured, dtype=torch.float32)
    assert (t >= lo).all() and (t <= hi).all(), (
        f"sampled slot indices out of allowed [{lo}, {hi}], min={t.min()}, max={t.max()}"
    )

    mean_obs = float(t.mean().item())
    std_obs = float(t.std(unbiased=True).item()) if len(captured) > 1 else 0.0

    # Monte Carlo slack for n≈100 on a wide bimodal discrete distribution.
    assert abs(mean_obs - mean_exp) < 55.0, (
        f"sample mean index {mean_obs:.2f} vs expected {mean_exp:.2f} "
        f"(n={len(captured)}, config means {mu1}/{mu2})"
    )
    assert abs(std_obs - std_exp) < 70.0, (
        f"sample std index {std_obs:.2f} vs expected {std_exp:.2f} (n={len(captured)})"
    )

    chi2 = _chi2_three_bins(captured, bin_probs)
    # df=2, conservative gate (fails if distribution is uniform or single-peaked wrong).
    assert chi2 < 25.0, f"chi-square vs binned theory too large: {chi2:.2f}"
