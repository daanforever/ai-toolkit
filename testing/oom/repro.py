"""Short Z-Image LoRA run for the CUDA memory-cap step-time problem.

Combat train shape: rank 64 attention LoRA, train dtype fp32, refiners on,
turbo_lora on, cuda_memory_fraction 0.92.
Sampling stays off and the step count stays short so the log is train steps.
Same `run_job` path as training, so `train.cuda_memory_fraction` is applied.
On Windows the default `max_split_size_mb:24` is set before torch is imported,
matching `run.py`. A normal pytest process cannot do that: `testing/conftest.py`
imports torch first. Launch this file, or the opt-in wrapper
`testing/oom/test_repro.py`.

Run from the repo root:
  venv\\Scripts\\python.exe testing/oom/repro.py

Env:
  OOM_CUDA_MEMORY_FRACTION   default 0.92; none/null/off passes null (no cap)
  OOM_REPRO_MAX_SPLIT        default 24 on Windows; off removes max_split_size_mb;
                             a number sets that size. An existing max_split_size_mb
                             is left in place unless this is off or a number.
  OOM_REPRO_STEPS            default 8
  OOM_REPRO_RESOLUTION       default 2048 (long side of the bucket grid)
  OOM_REPRO_DTYPE            default fp32 (train.dtype)
  OOM_REPRO_QUANTIZE         default off; float8 sets model.quantize and qtype float8
  ZIMAGE_DIFFSYNTH_MODEL_PATH
  ZIMAGE_DIFFSYNTH_SAMPLING_PATH
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

import gc
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch

from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (
    DEFAULT_ZIMAGE_MODEL_PATH,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
    ZImageDiffSynthModel,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
    ZImageDiffSynthTrainer,
)
from toolkit.util.debug import _cuda_storages
from toolkit.job import run_job

_DEFAULT_SAMPLING_PATH = (
    "e:/Backup/huggingface/hub/models--dimitribarbot--Z-Image-Turbo-BF16/"
    "snapshots/e11f18c054525642077b44c5edf00b99515a6182/"
)
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
    raw = os.environ.get("OOM_CUDA_MEMORY_FRACTION", "0.92").strip().lower()
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


def _quantize() -> tuple[bool, str]:
    raw = os.environ.get("OOM_REPRO_QUANTIZE", "off").strip().lower()
    if raw in ("off", "none", "null", "false", "0"):
        return False, "qfloat8"
    if raw == "float8":
        return True, "float8"
    raise SystemExit(
        f"[oom-repro] OOM_REPRO_QUANTIZE must be off or float8, got {raw!r}"
    )


def _dtype() -> str:
    raw = os.environ.get("OOM_REPRO_DTYPE", "fp32").strip().lower()
    if raw not in ("bf16", "fp16", "fp32"):
        raise SystemExit(
            f"[oom-repro] OOM_REPRO_DTYPE must be bf16, fp16, or fp32, got {raw!r}"
        )
    return raw


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


def _log_activation_census(latent: torch.Tensor, turbo: bool) -> None:
    """Large CUDA tensors still alive at the end of a grad-enabled forward."""
    claimed: set[int] = set()
    groups: dict[tuple, list[int]] = {}
    tracked = 0
    for obj in gc.get_objects():
        if not torch.is_tensor(obj) or obj.device.type != "cuda":
            continue
        shape = tuple(int(dim) for dim in obj.shape)
        logical = obj.numel() * obj.element_size()
        key = (str(obj.dtype), shape, bool(obj.requires_grad))
        for ptr, nbytes in _cuda_storages(obj):
            if ptr in claimed or logical != nbytes:
                continue
            claimed.add(ptr)
            tracked += nbytes
            group = groups.setdefault(key, [0, 0])
            group[0] += 1
            group[1] += nbytes
    alloc = torch.cuda.memory_allocated()
    _log(
        "[oom-repro] activations "
        f"turbo={int(turbo)} latent={_shape_of(latent)} "
        f"alloc_gib={_fmt_gib(alloc)} untracked_gib={_fmt_gib(alloc - tracked)}"
    )
    shown = 0
    for (dtype, shape, grad), (count, nbytes) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        if nbytes < 64 * 1024 * 1024:
            continue
        shown += 1
        _log(
            "[oom-repro] act "
            f"{dtype} {list(shape)} grad={str(grad).lower()} n={count} "
            f"gib={nbytes / (1024 ** 3):.3f}"
        )
    if shown == 0:
        _log("[oom-repro] act none")


def _shape_of(latent: torch.Tensor) -> str:
    if not torch.is_tensor(latent):
        return "?"
    return "x".join(str(int(dim)) for dim in latent.shape)


def _install_activation_probe() -> None:
    original = ZImageDiffSynthModel.get_noise_prediction

    def wrapped(self, latent_model_input, timestep, text_embeddings, *args, **kwargs):
        out = original(self, latent_model_input, timestep, text_embeddings, *args, **kwargs)
        if torch.is_grad_enabled():
            _log_activation_census(
                latent_model_input, bool(getattr(self, "_train_on_turbo", False))
            )
        return out

    ZImageDiffSynthModel.get_noise_prediction = wrapped


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
    sampling_path: str,
    steps: int,
    resolution: int,
    fraction,
    dtype: str,
    quantize: bool,
    qtype: str,
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
                        "rank_dropout": 0,
                        "type": "lora",
                        "linear": 64,
                        "linear_alpha": 64,
                        "conv": 0,
                        "conv_alpha": 0,
                        "lokr_full_rank": False,
                        "lokr_factor": -1,
                        "network_kwargs": {
                            "ignore_if_contains": [
                                "noise_refiner",
                                "context_refiner",
                                "all_final_layer",
                                "norm",
                            ],
                            "only_if_contains": [
                                "attention",
                                "to_q",
                                "to_k",
                                "to_v",
                            ],
                            "lora_down_init_scale": 1,
                        },
                    },
                    "save": {
                        "dtype": "fp32",
                        "save_every": steps,
                        "max_step_saves_to_keep": 1,
                        "save_format": "safetensors",
                        "push_to_hub": False,
                    },
                    "train": {
                        "lr": 1e-4,
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
                            "beta2": 0.996,
                            "weight_decay": 0.0001,
                            "scale_parameter": False,
                            "rms_max_decay_rate": 0.99,
                            "stochastic_accumulation": False,
                            "stochastic_rounding": False,
                            "factored": True,
                            "beta1": 0,
                        },
                        "unload_text_encoder": False,
                        "turbo_lora": True,
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
                    "logging": {"log_every": 1, "use_ui_logger": False, "debug": True},
                    "model": {
                        "dtype": "bf16",
                        "compile": False,
                        "debug_zimage_load": False,
                        "name_or_path": model_path,
                        "sampling_name_or_path": sampling_path,
                        "quantize": quantize,
                        "qtype": qtype,
                        "quantize_te": False,
                        "qtype_te": "qfloat8",
                        "arch": "zimage_diffsynth",
                        "low_vram": False,
                        "model_kwargs": {
                            "loader": "diffusers",
                            "use_dynamic_shifting": True,
                            "use_diffsynth_training_loop": False,
                            "disable_noise_refiner": False,
                            "disable_context_refiner": False,
                        },
                        "layer_offloading": False,
                        "layer_offloading_text_encoder_percent": 0,
                        "layer_offloading_transformer_percent": 0.1,
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
    sampling_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_SAMPLING_PATH", "").strip()
        or _DEFAULT_SAMPLING_PATH
    )
    if not sampling_path or not os.path.isdir(sampling_path):
        raise SystemExit(f"[oom-repro] Z-Image-Turbo snapshot is missing: {sampling_path}")

    steps = _positive_int("OOM_REPRO_STEPS", 8)
    resolution = _positive_int("OOM_REPRO_RESOLUTION", 2048)
    if resolution % 8 != 0:
        raise SystemExit(
            f"[oom-repro] OOM_REPRO_RESOLUTION must be a multiple of 8, got {resolution}"
        )
    dataset_dir = Path(_REPO_ROOT) / "datasets" / "chairs"
    if not dataset_dir.is_dir():
        raise SystemExit(f"[oom-repro] dataset is missing: {dataset_dir}")
    fraction = _fraction()
    dtype = _dtype()
    quantize, qtype = _quantize()
    props = torch.cuda.get_device_properties(0)
    _log(
        "[oom-repro] "
        f"device={props.name} total_gib={props.total_memory / (1024 ** 3):.2f} "
        f"fraction={fraction} alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')!r} "
        f"steps={steps} resolution={resolution} dtype={dtype} "
        f"quantize={quantize} qtype={qtype} "
        f"dataset={dataset_dir} model={model_path}"
    )

    work_root = Path(tempfile.gettempdir()) / "zimage_oom_repro"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    walls: list[float] = []
    _install_activation_probe()
    _install_probe(walls)
    run_job(
        _job_config(
            work_root,
            dataset_dir,
            model_path,
            sampling_path,
            steps,
            resolution,
            fraction,
            dtype,
            quantize,
            qtype,
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
