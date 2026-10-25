"""Short Z-Image LoRA run for the CUDA memory-cap step-time problem.

Four aspect ratios, rank 8, gradient checkpointing, refiners off, sampling off.
Same `run_job` path as training, so `train.cuda_memory_fraction` is applied.
On Windows the default `max_split_size_mb:24` is set before torch is imported,
matching `run.py`. A normal pytest process cannot do that: `testing/conftest.py`
imports torch first. Launch this file, or the opt-in wrapper
`testing/oom/test_repro.py`.

Run from the repo root:
  venv\\Scripts\\python.exe testing/oom/repro.py

Env:
  OOM_CUDA_MEMORY_FRACTION   default 0.95; none/null/off passes null (no cap)
  OOM_REPRO_MAX_SPLIT        default 24 on Windows; off removes max_split_size_mb;
                             a number sets that size. An existing max_split_size_mb
                             is left in place unless this is off or a number.
  OOM_REPRO_STEPS            default 8
  OOM_REPRO_RESOLUTION       default 1024 (long side of the bucket grid)
  OOM_REPRO_DTYPE            default bf16 (train.dtype)
  ZIMAGE_DIFFSYNTH_MODEL_PATH
"""

from __future__ import annotations

import os
import sys


def _apply_alloc_conf() -> None:
    """Mirror run.py, with an explicit off switch for A/B runs."""
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    parts = [part for part in current.split(",") if part]
    kept = [part for part in parts if not part.startswith("max_split_size_mb")]
    raw = os.environ.get("OOM_REPRO_MAX_SPLIT", "").strip().lower()
    if raw in ("off", "none", "null"):
        _write_alloc_conf(kept)
        return
    if raw == "":
        if os.name != "nt" or any(part.startswith("max_split_size_mb") for part in parts):
            return
        size = "24"
    else:
        if not raw.isdigit() or int(raw) <= 0:
            raise SystemExit(
                f"[oom-repro] OOM_REPRO_MAX_SPLIT must be off or a positive integer, got {raw!r}"
            )
        size = raw
    _write_alloc_conf(kept + [f"max_split_size_mb:{size}"])


def _write_alloc_conf(parts: list[str]) -> None:
    if parts:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


_apply_alloc_conf()

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if sys.platform == "win32":
    _venv_python = os.path.join(_REPO_ROOT, "venv", "Scripts", "python.exe")
else:
    _venv_python = os.path.join(_REPO_ROOT, "venv", "bin", "python")
if os.path.isfile(_venv_python):
    if os.path.realpath(sys.executable) != os.path.realpath(_venv_python):
        os.execv(_venv_python, [_venv_python] + sys.argv[1:])

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if "torch" in sys.modules:
    raise SystemExit(
        "[oom-repro] torch is already imported; PYTORCH_CUDA_ALLOC_CONF would not apply. "
        "Run this file in a fresh interpreter."
    )

import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image

from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (
    DEFAULT_ZIMAGE_MODEL_PATH,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
    ZImageDiffSynthTrainer,
)
from toolkit.job import run_job

_ASPECTS = ((1024, 1024), (768, 1344), (1344, 768), (896, 1152))
_STAT_KEYS = (
    ("retries", "num_alloc_retries"),
    ("syncs", "num_sync_all_streams"),
    ("mallocs", "num_device_alloc"),
    ("frees", "num_device_free"),
    ("oversize_seg", "oversize_segments.current"),
    ("oversize_alloc", "oversize_allocations.current"),
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    if not raw.isdigit() or int(raw) <= 0:
        raise SystemExit(f"[oom-repro] {name} must be a positive integer, got {raw!r}")
    return int(raw)


def _fraction():
    raw = os.environ.get("OOM_CUDA_MEMORY_FRACTION", "0.95").strip().lower()
    if raw in ("none", "null", "off"):
        return None
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(
            f"[oom-repro] OOM_CUDA_MEMORY_FRACTION must be a float in (0, 1] or none, got {raw!r}"
        ) from None
    if not (0.0 < value <= 1.0):
        raise SystemExit(
            f"[oom-repro] OOM_CUDA_MEMORY_FRACTION must be a float in (0, 1] or none, got {raw!r}"
        )
    return value


def _dtype() -> str:
    raw = os.environ.get("OOM_REPRO_DTYPE", "bf16").strip().lower()
    if raw not in ("bf16", "fp16", "fp32"):
        raise SystemExit(
            f"[oom-repro] OOM_REPRO_DTYPE must be bf16, fp16, or fp32, got {raw!r}"
        )
    return raw


def _pixel_sizes(resolution: int) -> list[tuple[int, int]]:
    sizes = []
    for width, height in _ASPECTS:
        snapped_w = max(8, (width * resolution // 1024) // 8 * 8)
        snapped_h = max(8, (height * resolution // 1024) // 8 * 8)
        sizes.append((snapped_w, snapped_h))
    return sizes


def _write_dataset(dataset_dir: Path, resolution: int) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for index, (width, height) in enumerate(_pixel_sizes(resolution)):
        Image.new("RGB", (width, height), (32 + index * 40, 64, 128)).save(
            dataset_dir / f"{index:03d}.png"
        )
        (dataset_dir / f"{index:03d}.txt").write_text("dog", encoding="utf-8")


def _latent_shape(batch) -> str:
    item = batch[0] if isinstance(batch, list) else batch
    latents = getattr(item, "latents", None)
    if isinstance(latents, list):
        latents = latents[0] if latents else None
    if latents is None or not hasattr(latents, "shape"):
        return "?"
    return "x".join(str(int(dim)) for dim in latents.shape)


def _snap() -> dict:
    stats = torch.cuda.memory_stats()
    snap = {
        "alloc": stats.get("allocated_bytes.all.current"),
        "reserved": stats.get("reserved_bytes.all.current"),
    }
    for name, key in _STAT_KEYS:
        snap[name] = stats.get(key)
    return snap


def _fmt_count(before, after) -> str:
    if before is None or after is None:
        return "na"
    return str(int(after) - int(before))


def _fmt_gib(value) -> str:
    if value is None:
        return "na"
    return f"{value / (1024 ** 3):.2f}"


def _install_probe(walls: list[float]) -> None:
    original = ZImageDiffSynthTrainer.hook_train_loop

    def wrapped(self, batch):
        before = _snap()
        started = time.perf_counter()
        try:
            return original(self, batch)
        finally:
            elapsed = time.perf_counter() - started
            walls.append(elapsed)
            after = _snap()
            _log(
                "[oom-repro] "
                f"step={len(walls)} wall_s={elapsed:.3f} latent={_latent_shape(batch)} "
                f"alloc_gib={_fmt_gib(after['alloc'])} reserved_gib={_fmt_gib(after['reserved'])} "
                f"d_retry={_fmt_count(before['retries'], after['retries'])} "
                f"d_sync={_fmt_count(before['syncs'], after['syncs'])} "
                f"d_malloc={_fmt_count(before['mallocs'], after['mallocs'])} "
                f"d_free={_fmt_count(before['frees'], after['frees'])} "
                f"oversize_seg={after['oversize_seg']} oversize_alloc={after['oversize_alloc']}"
            )

    ZImageDiffSynthTrainer.hook_train_loop = wrapped


def _job_config(
    work_root: Path,
    dataset_dir: Path,
    model_path: str,
    steps: int,
    resolution: int,
    fraction,
    dtype: str,
) -> dict:
    return {
        "job": "extension",
        "config": {
            "name": "zimage_oom_repro",
            "process": [
                {
                    "type": "z_image_diffsynth_trainer",
                    "log_dir": str(work_root / "TensorBoard"),
                    "training_folder": str(work_root / "output"),
                    "sqlite_db_path": str(work_root / "aitk_db.db"),
                    "device": "cuda",
                    "trigger_word": None,
                    "performance_log_every": 0,
                    "network": {
                        "pretrained_lora_path": "",
                        "dtype": "fp32",
                        "rank_dropout": 0.01,
                        "type": "lora",
                        "linear": 8,
                        "linear_alpha": 8,
                        "conv": 0,
                        "conv_alpha": 0,
                        "lokr_full_rank": False,
                        "lokr_factor": -1,
                        "network_kwargs": {
                            "ignore_if_contains": [
                                "noise_refiner",
                                "context_refiner",
                                "all_final_layer",
                            ],
                            "lora_down_init_scale": 1,
                        },
                    },
                    "save": {
                        "dtype": "bf16",
                        "save_every": steps,
                        "max_step_saves_to_keep": 1,
                        "save_format": "safetensors",
                        "push_to_hub": False,
                    },
                    "train": {
                        "lr": 1e-4,
                        "noise_offset": 0.1,
                        "batch_size": 1,
                        "gradient_accumulation": 1,
                        "noise_scheduler": "flowmatch",
                        "prediction_type": "flowmatch",
                        "timestep_type": "linear",
                        "content_or_style": "balanced",
                        "bypass_guidance_embedding": False,
                        "steps": steps,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "optimizer": "adafactor",
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
                        "disable_sampling": True,
                        "cuda_memory_fraction": fraction,
                        "dtype": dtype,
                        "diff_output_preservation": False,
                        "min_snr_gamma": 0,
                        "loss_type": "mse",
                    },
                    "logging": {"log_every": 1, "use_ui_logger": False, "debug": False},
                    "model": {
                        "dtype": "bf16",
                        "compile": False,
                        "debug_zimage_load": False,
                        "name_or_path": model_path,
                        "quantize": False,
                        "qtype": "qfloat8",
                        "quantize_te": False,
                        "qtype_te": "qfloat8",
                        "arch": "zimage_diffsynth",
                        "low_vram": False,
                        "model_kwargs": {
                            "loader": "diffusers",
                            "use_dynamic_shifting": False,
                            "use_diffsynth_training_loop": False,
                            "disable_noise_refiner": True,
                            "disable_context_refiner": True,
                        },
                        "layer_offloading": False,
                        "layer_offloading_text_encoder_percent": 1,
                        "layer_offloading_transformer_percent": 1,
                    },
                    "datasets": [
                        {
                            "folder_path": str(dataset_dir),
                            "square_crop": False,
                            "buckets": True,
                            "shuffle_tokens": False,
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0,
                            "cache_latents_to_disk": True,
                            "is_reg": False,
                            "network_weight": 1,
                            "resolution": [resolution],
                            "controls": [],
                            "flip_x": False,
                            "flip_y": False,
                            "num_repeats": 1,
                        }
                    ],
                }
            ],
        },
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("[oom-repro] CUDA is not available")
    model_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip()
        or DEFAULT_ZIMAGE_MODEL_PATH
    )
    if not model_path or not os.path.isdir(model_path):
        raise SystemExit(f"[oom-repro] Z-Image snapshot is missing: {model_path}")

    steps = _positive_int("OOM_REPRO_STEPS", 8)
    resolution = _positive_int("OOM_REPRO_RESOLUTION", 1024)
    if resolution % 8 != 0:
        raise SystemExit(
            f"[oom-repro] OOM_REPRO_RESOLUTION must be a multiple of 8, got {resolution}"
        )
    fraction = _fraction()
    dtype = _dtype()
    props = torch.cuda.get_device_properties(0)
    _log(
        "[oom-repro] "
        f"device={props.name} total_gib={props.total_memory / (1024 ** 3):.2f} "
        f"fraction={fraction} alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')!r} "
        f"steps={steps} resolution={resolution} dtype={dtype} model={model_path}"
    )

    work_root = Path(tempfile.gettempdir()) / "zimage_oom_repro"
    if work_root.exists():
        shutil.rmtree(work_root)
    dataset_dir = work_root / "datasets" / "1"
    _write_dataset(dataset_dir, resolution)
    walls: list[float] = []
    _install_probe(walls)
    run_job(
        _job_config(
            work_root,
            dataset_dir,
            model_path,
            steps,
            resolution,
            fraction,
            dtype,
        )
    )
    if len(walls) >= 2:
        later = walls[1:]
        _log(
            "[oom-repro] "
            f"steps={len(walls)} first_wall_s={walls[0]:.3f} "
            f"later_median_s={statistics.median(later):.3f}"
        )
    elif walls:
        _log(f"[oom-repro] steps=1 wall_s={walls[0]:.3f}")
    else:
        raise SystemExit("[oom-repro] run_job finished without a training step")
    _log("[oom-repro] done")


if __name__ == "__main__":
    main()
