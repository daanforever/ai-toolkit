"""One real Z-Image + Z-Image-Turbo optimizer step with turbo_lora: true.

Skipped unless ZIMAGE_TURBO_LORA_GPU=1, CUDA is available, and both snapshot
directories exist. A normal pytest run does not load the models.

Run from the repo root:
  venv\\Scripts\\python.exe -m extensions_built_in.diffusion_models.z_image_diffsynth.turbo_lora.test_gpu
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_OPT_IN = "ZIMAGE_TURBO_LORA_GPU"
_STARTUP = "[zimage_diffsynth] turbo_lora=true weight=1 slots=8"


def _opted_in() -> bool:
    return os.environ.get(_OPT_IN) == "1"


pytestmark = pytest.mark.skipif(
    not _opted_in(),
    reason="set ZIMAGE_TURBO_LORA_GPU=1 to train on real Z-Image and Z-Image-Turbo",
)


def _snapshot_paths() -> tuple[str, str]:
    from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (
        DEFAULT_ZIMAGE_MODEL_PATH,
        DEFAULT_ZIMAGE_SAMPLING_PATH,
    )

    model_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip()
        or DEFAULT_ZIMAGE_MODEL_PATH
    )
    sampling_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_SAMPLING_PATH", "").strip()
        or DEFAULT_ZIMAGE_SAMPLING_PATH
    )
    return model_path, sampling_path


def _write_dataset(dataset_dir: Path) -> None:
    from PIL import Image

    dataset_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), (32, 64, 128)).save(dataset_dir / "000.png")
    (dataset_dir / "000.txt").write_text("dog", encoding="utf-8")


def _job_config(
    train_name: str,
    output_root: Path,
    work_root: Path,
    dataset_dir: Path,
    model_path: str,
    sampling_path: str,
) -> dict:
    return {
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
                        "linear": 8,
                        "linear_alpha": 8,
                        "conv": 0,
                        "conv_alpha": 0,
                        "lokr_full_rank": False,
                        "lokr_factor": -1,
                        "network_kwargs": {
                            "ignore_if_contains": [],
                            "lora_down_init_scale": 1,
                        },
                        "pretrained_lora_path": "",
                    },
                    "save": {
                        "dtype": "bf16",
                        "save_every": 1,
                        "max_step_saves_to_keep": 2,
                        "save_format": "safetensors",
                        "push_to_hub": False,
                    },
                    "train": {
                        "lr": 0.000025,
                        "noise_offset": 0.1,
                        "max_denoising_steps": 995,
                        "min_denoising_steps": 5,
                        "batch_size": 1,
                        "bypass_guidance_embedding": False,
                        "steps": 1,
                        "gradient_accumulation": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adafactor",
                        "timestep_type": "linear",
                        "content_or_style": "gaussian",
                        "gaussian_mean": 450,
                        "gaussian_std": 0.45,
                        "optimizer_params": {
                            "beta2": 0,
                            "weight_decay": 0.01,
                            "scale_parameter": False,
                            "relative_step": True,
                            "warmup_init": True,
                            "warmup_steps": 20,
                            "rms_max_decay_rate": 0.99,
                            "stochastic_accumulation": True,
                            "stochastic_rounding": True,
                            "factored": True,
                            "beta1": 0.9,
                        },
                        "unload_text_encoder": True,
                        "cache_text_embeddings": True,
                        "turbo_lora": True,
                        "ema_config": {"use_ema": False, "ema_decay": 0.99},
                        "skip_first_sample": True,
                        "force_first_sample": False,
                        "disable_sampling": True,
                        "dtype": "bf16",
                        "diff_output_preservation": False,
                        "diff_output_preservation_multiplier": 1,
                        "diff_output_preservation_class": "person",
                        "switch_boundary_every": 1,
                        "loss_type": "mse",
                        "blank_prompt_preservation": False,
                        "blank_prompt_probability": 0,
                        "blank_prompt_preservation_multiplier": 0.5,
                        "min_snr_gamma": 7,
                    },
                    "logging": {"log_every": 1, "use_ui_logger": True, "debug": True},
                    "model": {
                        "debug_zimage_load": False,
                        "name_or_path": model_path,
                        "sampling_name_or_path": sampling_path,
                        "dtype": "bf16",
                        "quantize": False,
                        "qtype": "qfloat8",
                        "quantize_te": False,
                        "qtype_te": "qfloat8",
                        "arch": "zimage_diffsynth",
                        "low_vram": False,
                        "model_kwargs": {
                            "use_diffsynth_training_loop": False,
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
                            "caption_dropout_rate": 0,
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
                        "guidance_scale": 1,
                        "sample_steps": 9,
                        "num_frames": 1,
                        "fps": 1,
                    },
                }
            ],
        },
    }


def test_turbo_lora_gpu_train(capsys):
    import torch
    from toolkit.job import run_job

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    model_path, sampling_path = _snapshot_paths()
    if not model_path or not os.path.isdir(model_path):
        pytest.skip(f"Z-Image snapshot is missing: {model_path}")
    if not sampling_path or not os.path.isdir(sampling_path):
        pytest.skip(f"Z-Image-Turbo snapshot is missing: {sampling_path}")

    work_root = Path(tempfile.gettempdir()) / "zimage_diffsynth_turbo_lora_gpu"
    if work_root.exists():
        shutil.rmtree(work_root)
    dataset_dir = work_root / "datasets" / "1"
    _write_dataset(dataset_dir)
    output_root = work_root / "output"
    output_root.mkdir(parents=True)
    train_name = "zimage_diffsynth_turbo_lora_gpu"
    run_job(
        _job_config(
            train_name,
            output_root,
            work_root,
            dataset_dir,
            model_path,
            sampling_path,
        )
    )
    captured = capsys.readouterr()
    assert _STARTUP in captured.out
    saved = list(output_root.rglob("*.safetensors"))
    assert saved, f"no checkpoint under {output_root}"


if __name__ == "__main__":
    os.environ[_OPT_IN] = "1"
    raise SystemExit(pytest.main([__file__, "-q"]))
