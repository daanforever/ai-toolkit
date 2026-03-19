"""
Train smoke test for Z-Image DiffSynth LoRA.

Scenario:
1) Create a LoRA network config.
2) Generate 10 images for prompt "dog" with different seeds (save first seed).
3) Train LoRA on those images for 10 epochs equivalent.
4) Generate again with trained LoRA using the saved first seed.
5) Compare cosine similarity between before/after images.

Source images + captions are stored under repo ``temp/test_train/`` (``000.png`` … ``009.png``,
matching ``.txt`` files, plus ``test_train_manifest.json``). If that cache is valid, step 1 is
skipped — no full model / text encoder load for dataset creation (``run_job`` still loads its
own model for training).

Env:
  ZIMAGE_TEST_TRAIN_FORCE_REGEN=1 — drop cache and regenerate source images.
  ZIMAGE_DIFFSYNTH_DEBUG — see toolkit debug (existing).

Why step 3 used to die with exit code 1 (often no Python traceback):

- After ``run_job``, ``torch.cuda.memory_allocated()`` stayed ~9 GB while ``mem_get_info`` free
  ~5–6 GB — not enough to load the Z-Image stack again → OOM / native abort during checkpoint load.
- Contributing factors: Hugging Face ``Accelerator`` singleton (``toolkit.accelerator``), the
  trainer process holding ``accelerator`` / cached embeds / prepared modules, and CUDA allocator
  state in-process. Clearing globals + nulling attributes on the job process (see
  ``toolkit.job.run_job``) and ``gc``/``empty_cache`` still left ~9 GB allocated here.
- **Fix:** training (step 2) and post-train generate (step 3) each run in a **fresh Python
  subprocess** by default so the parent never holds training VRAM while a child loads the model
  (otherwise the driver can abort with e.g. Windows exit ``3221225477`` / access violation).
  Set ``ZIMAGE_TEST_TRAIN_INPROCESS_TRAIN=1`` / ``ZIMAGE_TEST_TRAIN_INPROCESS_POST=1`` to force
  in-process steps (debug / profiler only).

Run from repo root:
  python -m extensions_built_in.diffusion_models.z_image_diffsynth.test_train
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import gc
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Re-run with venv Python if venv exists and we're not already using it
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

# repo root on path
TOOLKIT_ROOT = _REPO_ROOT
if TOOLKIT_ROOT not in sys.path:
    sys.path.insert(0, TOOLKIT_ROOT)

# Enable toolkit debug / memory_debug output by default so this test always
# prints CUDA / RAM usage for key steps. Can be overridden by env var.
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

from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.job import run_job
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.unloader import FakeTextEncoder, unload_text_encoder
from toolkit.util.get_model import get_model_class
from toolkit.util.debug import memory_debug
from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (
    DEFAULT_ZIMAGE_MODEL_PATH,
    DEFAULT_ZIMAGE_SAMPLING_PATH,
)

NUM_SOURCE_IMAGES = 10
TEST_TRAIN_IMAGE_CACHE = Path(TOOLKIT_ROOT) / "temp" / "test_train"
TEST_TRAIN_MANIFEST_NAME = "test_train_manifest.json"
LORA_PATH_MARKER_NAME = ".zimage_test_train_lora.txt"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _cuda_mem_log(tag: str) -> None:
    if not torch.cuda.is_available():
        _log(f"[DEBUG CUDA {tag}] CUDA not available")
        return
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        _log(
            f"[DEBUG CUDA {tag}] free={free_b / 1e9:.2f} GB | "
            f"total={total_b / 1e9:.2f} GB | alloc={alloc / 1e9:.2f} GB | "
            f"reserved={reserved / 1e9:.2f} GB"
        )
    except Exception as exc:
        _log(f"[DEBUG CUDA {tag}] mem_get_info failed: {exc!r}")


def _expected_manifest(prompt: str, seeds: list[int]) -> dict:
    return {
        "version": 1,
        "prompt": prompt,
        "seeds": seeds,
        "num_images": NUM_SOURCE_IMAGES,
    }


def _cache_paths(cache_dir: Path) -> tuple[list[Path], list[Path]]:
    pngs = [cache_dir / f"{i:03d}.png" for i in range(NUM_SOURCE_IMAGES)]
    txts = [cache_dir / f"{i:03d}.txt" for i in range(NUM_SOURCE_IMAGES)]
    return pngs, txts


def _is_image_cache_valid(cache_dir: Path, prompt: str, seeds: list[int]) -> bool:
    manifest_path = cache_dir / TEST_TRAIN_MANIFEST_NAME
    if not manifest_path.is_file():
        _log(f"[DEBUG cache] no manifest at {manifest_path}")
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log(f"[DEBUG cache] manifest read/JSON failed: {exc!r}")
        return False
    exp = _expected_manifest(prompt, seeds)
    if data.get("version") != exp["version"]:
        _log(f"[DEBUG cache] manifest version mismatch: {data.get('version')} != {exp['version']}")
        return False
    if data.get("prompt") != exp["prompt"] or data.get("seeds") != exp["seeds"]:
        _log("[DEBUG cache] manifest prompt/seeds mismatch; delete cache or set ZIMAGE_TEST_TRAIN_FORCE_REGEN=1")
        return False
    if data.get("num_images") != NUM_SOURCE_IMAGES:
        _log(f"[DEBUG cache] num_images mismatch: {data.get('num_images')}")
        return False
    pngs, txts = _cache_paths(cache_dir)
    for p in pngs + txts:
        if not p.is_file():
            _log(f"[DEBUG cache] missing file: {p}")
            return False
        if p.stat().st_size <= 0:
            _log(f"[DEBUG cache] empty file: {p}")
            return False
    _log(f"[DEBUG cache] valid: {cache_dir} ({NUM_SOURCE_IMAGES} png+txt)")
    return True


def _write_manifest(cache_dir: Path, prompt: str, seeds: list[int]) -> None:
    manifest_path = cache_dir / TEST_TRAIN_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(_expected_manifest(prompt, seeds), indent=2),
        encoding="utf-8",
    )
    _log(f"[DEBUG cache] wrote manifest {manifest_path}")


def _populate_dataset_from_cache(cache_dir: Path, dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pngs, txts = _cache_paths(cache_dir)
    for i in range(NUM_SOURCE_IMAGES):
        shutil.copy2(pngs[i], dataset_dir / pngs[i].name)
        shutil.copy2(txts[i], dataset_dir / txts[i].name)
    _log(f"[DEBUG cache] copied {NUM_SOURCE_IMAGES} image+caption pairs -> {dataset_dir}")


def _release_cuda(tag: str) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    _log(f"[MEM] released CUDA cache ({tag})")


def _aggressive_cuda_release(tag: str) -> None:
    """Free GPU memory after run_job: job/process graphs may need several gc passes."""
    for _ in range(4):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
    _log(f"[MEM] aggressive CUDA+gc release ({tag})")
    _cuda_mem_log(tag)


def _drop_accelerate_global_singleton(tag: str) -> None:
    """Train path uses get_accelerator() singleton; prepare() keeps module refs until this is cleared."""
    try:
        import toolkit.accelerator as acc
    except Exception as exc:
        _log(f"[DEBUG accelerate] import failed: {exc!r}")
        return
    had = getattr(acc, "global_accelerator", None) is not None
    acc.global_accelerator = None
    _log(f"[DEBUG accelerate] global_accelerator cleared ({tag}); had_singleton={had}")
    gc.collect()


def _log_cuda_memory_diagnostic(heading: str) -> None:
    if not torch.cuda.is_available():
        return
    _log(f"[DEBUG CUDA diagnostic] {heading}")
    try:
        _log(torch.cuda.memory_summary(device=0, abbreviated=False))
    except Exception as exc:
        _log(f"[DEBUG CUDA diagnostic] memory_summary failed: {exc!r}")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "True", "yes")


def _train_subprocess_worker() -> None:
    """Run ``run_job`` training only; write absolute LoRA path to marker file in work_root."""
    _log("[train subprocess] worker start")
    work_root = Path(os.environ["ZIMAGE_TEST_TRAIN_WORK_ROOT"])
    dataset_dir = Path(os.environ["ZIMAGE_TEST_TRAIN_DATASET_DIR"])
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
    lora_path = _train_lora(work_root, dataset_dir, model_path, sampling_path)
    marker = work_root / LORA_PATH_MARKER_NAME
    marker.write_text(str(lora_path.resolve()), encoding="utf-8")
    _log(f"[train subprocess] marker -> {marker}")


def _post_train_subprocess_worker() -> None:
    """Fresh interpreter: load base model + LoRA and write ``after_lora_seed0.png``."""
    _log("[post-train subprocess] worker start")
    device = torch.device("cuda")
    lora_path = os.environ["ZIMAGE_TEST_TRAIN_LORA_PATH"]
    work_root = Path(os.environ["ZIMAGE_TEST_TRAIN_WORK_ROOT"])
    first_seed = int(os.environ["ZIMAGE_TEST_TRAIN_FIRST_SEED"])
    prompt = os.environ.get("ZIMAGE_TEST_TRAIN_PROMPT", "dog")
    after_path = work_root / "after_lora_seed0.png"
    _cuda_mem_log("subprocess worker initial")
    try:
        after_model = _build_model(device, phase="POST_TRAIN")
        _attach_lora_for_inference(after_model, lora_path)
        after_pipeline, after_cond, after_uncond = _prepare_generation(after_model, prompt)
        _generate_images_batch(
            after_model,
            after_pipeline,
            after_cond,
            after_uncond,
            prompt,
            [first_seed],
            [after_path],
            mem_debug_tag="test_train after_lora generate_images (subprocess)",
        )
        _log(f"[post-train subprocess] wrote {after_path}")
    except torch.cuda.OutOfMemoryError:
        _log("[post-train subprocess] CUDA OOM")
        traceback.print_exc()
        _log_cuda_memory_diagnostic("subprocess OOM")
        raise
    except Exception:
        traceback.print_exc()
        _log_cuda_memory_diagnostic("subprocess exception")
        raise


def _offload_text_encoder(sd) -> None:
    try:
        # Same as SDTrainer when cache_text_embeddings=True: unload_text_encoder replaces
        # model.text_encoder with FakeTextEncoder; training uses cached disk embeds only.
        unload_text_encoder(sd)
        _release_cuda("text_encoder_offload")
    except Exception:
        pass


def _point_pipeline_te_at_model_placeholder(sd, pipeline) -> None:
    """Diffusers pipeline still holds its own TE after get_generation_pipeline; swap to model's Fake.

    generate_single_image passes prompt_embeds with prompt=None, so encode_prompt / TE forward
    is not used. This drops the duplicate real TE from the pipeline reference (like training
    after unload) and matches cache_text_embeddings semantics in the test only.
    """
    if pipeline is None or not hasattr(pipeline, "text_encoder"):
        return
    te = getattr(sd, "text_encoder", None)
    if isinstance(te, list):
        te = te[0] if te else None
    if isinstance(te, FakeTextEncoder):
        pipeline.text_encoder = te


def _image_to_tensor(image) -> torch.Tensor:
    arr = np.array(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().float()
    bf = b.flatten().float()
    denom = torch.norm(af) * torch.norm(bf)
    if denom.item() == 0:
        return 0.0
    return float(torch.dot(af, bf) / denom)


def _build_model(device: torch.device, phase: str = "unknown"):
    model_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip()
        or DEFAULT_ZIMAGE_MODEL_PATH
    )
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(
            "Z-Image model path is missing or invalid. "
            "Set ZIMAGE_DIFFSYNTH_MODEL_PATH or update DEFAULT_ZIMAGE_MODEL_PATH."
        )

    sampling_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_SAMPLING_PATH", "").strip()
        or DEFAULT_ZIMAGE_SAMPLING_PATH
        or None
    )
    if sampling_path and not os.path.isdir(sampling_path):
        sampling_path = None

    model_cfg = {
        "name_or_path": model_path,
        "arch": "zimage_diffsynth",
        "quantize": True,
        "qtype": "qfloat8",
        "quantize_te": True,
        "qtype_te": "qfloat8",
        "model_kwargs": {
            "use_diffsynth_training_loop": False,
            "disable_noise_refiner": False,
            "disable_context_refiner": False,
            "sampling_loader": "diffusers",
        },
    }
    if sampling_path:
        model_cfg["sampling_name_or_path"] = sampling_path

    _log(f"[PHASE {phase}] build_model: start")
    with memory_debug(_log, f"test_train build_model ({phase})", kind="cuda"):
        model_config = ModelConfig(**model_cfg)
        model_cls = get_model_class(model_config)
        sd = model_cls(device, model_config, dtype="bf16")
        sd.load_model()
    _log(f"[PHASE {phase}] build_model: done")
    return sd


def _prepare_generation(sd, prompt: str):
    _log("[PHASE PREP] get_pipeline: start")
    with memory_debug(_log, "test_train prepare_generation get_pipeline", kind="cuda"):
        pipeline = sd.get_generation_pipeline()
    _log("[PHASE PREP] get_pipeline: done")
    _log("[PHASE PREP] get_prompt_embeds: start")
    with memory_debug(_log, "test_train prepare_generation get_prompt_embeds", kind="cuda"):
        cond = sd.get_prompt_embeds(prompt)
        uncond = sd.get_prompt_embeds("")
    _log("[PHASE PREP] get_prompt_embeds: done")
    _offload_text_encoder(sd)
    _point_pipeline_te_at_model_placeholder(sd, pipeline)
    return pipeline, cond, uncond


def _generate_images_batch(
    sd,
    pipeline,
    cond,
    uncond,
    prompt: str,
    seeds: list[int],
    out_paths: list[Path],
    max_sec_per_image: float = 20.0,
    mem_debug_tag: str = "test_train generate_images batch",
) -> None:
    """Use BaseModel.generate_images (batch sampling path), not repeated generate_single_image.

    z_image_diffsynth only sets _sampling_in_batch_generate inside generate_images; otherwise
    each image pays standalone main↔GPU moves and breaks reasonable per-image wall time.
    Embeddings come from sample_prompts_cache (TE is Fake after prepare); pipeline must be
    the one from prepare — we patch get_generation_pipeline because ZImageDiffSynthModel.generate_images
    does not accept a pipeline argument.
    """
    n = len(seeds)
    if len(out_paths) != n:
        raise ValueError("seeds and out_paths length mismatch")
    sd.sample_prompts_cache = [
        {"conditional": cond, "unconditional": uncond} for _ in range(n)
    ]
    configs = [
        GenerateImageConfig(
            width=1024,
            height=768,
            num_inference_steps=9,
            guidance_scale=1.0,
            prompt=prompt,
            negative_prompt="",
            output_path=str(out_paths[i]),
            seed=seeds[i],
        )
        for i in range(n)
    ]
    orig_ggp = sd.get_generation_pipeline
    try:
        try:
            pipeline.set_progress_bar_config(disable=True)
        except Exception:
            pass

        def _return_prepared_pipeline():
            return pipeline

        sd.get_generation_pipeline = _return_prepared_pipeline  # type: ignore[method-assign]
        started_at = time.perf_counter()
        with memory_debug(_log, mem_debug_tag, kind="cuda"):
            sd.generate_images(configs)
        elapsed = time.perf_counter() - started_at
    finally:
        sd.get_generation_pipeline = orig_ggp  # type: ignore[method-assign]
        sd.sample_prompts_cache = None
    per = elapsed / max(n, 1)
    if per > max_sec_per_image:
        raise TimeoutError(
            f"Sampling too slow: {elapsed:.2f}s for {n} image(s) "
            f"({per:.2f}s/image > {max_sec_per_image:.2f}s/image)"
        )


def _attach_lora_for_inference(sd, lora_path: str) -> None:
    network_config = NetworkConfig(
        type="lora",
        linear=128,
        linear_alpha=128,
        conv=0,
        conv_alpha=0,
        rank_dropout=0.01,
        pretrained_lora_path=lora_path,
        network_kwargs={
            "ignore_if_contains": [],
            "lora_down_init_scale": 1,
            "init_lora_weights": "pissa",
        },
    )
    network_kwargs = dict(network_config.network_kwargs or {})
    if hasattr(sd, "target_lora_modules"):
        network_kwargs["target_lin_modules"] = sd.target_lora_modules

    common = dict(
        text_encoder=sd.text_encoder,
        lora_dim=network_config.linear,
        multiplier=1.0,
        alpha=network_config.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        conv_lora_dim=network_config.conv,
        conv_alpha=network_config.conv_alpha,
        is_sdxl=False,
        is_v2=False,
        is_v3=False,
        is_pixart=False,
        is_auraflow=False,
        is_flux=False,
        is_lumina2=False,
        is_ssd=False,
        is_vega=False,
        dropout=network_config.dropout,
        rank_dropout=network_config.rank_dropout,
        module_dropout=network_config.module_dropout,
        use_text_encoder_1=True,
        use_text_encoder_2=True,
        use_bias=False,
        is_lorm=False,
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        is_transformer=getattr(sd, "is_transformer", True),
        base_model=sd,
        **network_kwargs,
    )

    with memory_debug(_log, "test_train attach_lora main_network", kind="cuda"):
        main_network = LoRASpecialNetwork(
            unet=sd.get_model_to_train(),
            **common,
        )
        main_network.force_to(sd.device_torch, dtype=torch.float32)
        main_network.apply_to(
            sd.text_encoder,
            sd.unet,
            train_text_encoder=False,
            train_unet=True,
        )
        main_network.load_weights(lora_path)
        main_network.multiplier = 1.0
        main_network._update_torch_multiplier()
        sd.network = main_network

    if getattr(sd, "_sampling_transformer", None) is not None:
        with memory_debug(_log, "test_train attach_lora sampling_network", kind="cuda"):
            sampling_network = LoRASpecialNetwork(
                unet=sd._sampling_transformer,
                **common,
            )
            sampling_network.share_parameters_with(main_network)
            sampling_network.apply_to(
                sd.text_encoder,
                sd._sampling_transformer,
                train_text_encoder=False,
                train_unet=True,
            )
            sampling_network.multiplier = 1.0
            sampling_network._update_torch_multiplier()
            sd._sampling_network = sampling_network


def _train_lora(work_root: Path, dataset_dir: Path, model_path: str, sampling_path: str | None) -> Path:
    train_name = "zimage_diffsynth_train_smoke"
    output_root = work_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    # Keep gradient_accumulation=8 as in the provided config and map
    # "10 epochs on 10 images" (~100 batch iterations) to optimizer steps.
    total_steps = 13

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
                        "linear": 128,
                        "linear_alpha": 128,
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
                        "save_every": 10,
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
                        "steps": total_steps,
                        "gradient_accumulation": 8,
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
                            "emergency_brake": 0.75,
                            "beta2": 0.9,
                            "weight_decay": 0.01,
                            "scale_parameter": False,
                            "relative_step": True,
                            "warmup_init": True,
                            "warmup_steps": 20,
                            "min_lr": 0,
                            "rms_max_decay_rate": 0.99,
                            "stochastic_accumulation": True,
                            "stochastic_rounding": True,
                            "factored": True,
                            "beta1": 0.9,
                            "saddle_point_window": 100,
                            "saddle_point_threshold": 0.001,
                            "saddle_point_step": 0.01,
                        },
                        # Matches SDTrainer: cache to disk, then unload_text_encoder -> Fake TE; no live TE in train loop.
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
                        "min_snr_gamma": 7,
                    },
                    "logging": {"log_every": 1, "use_ui_logger": True, "debug": True},
                    "model": {
                        "debug_zimage_load": False,
                        "name_or_path": model_path,
                        "sampling_name_or_path": sampling_path,
                        "quantize": True,
                        "qtype": "qfloat8",
                        "quantize_te": True,
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
            ],
        },
    }

    with memory_debug(_log, "test_train run_job train_lora", kind="cuda"):
        _log("[PHASE TRAIN] run_job: start")
        run_job(config)
        _log("[PHASE TRAIN] run_job: done")

    del config
    _cuda_mem_log("after run_job, before drop Accelerate singleton")
    _drop_accelerate_global_singleton("after run_job")
    _aggressive_cuda_release("after run_job (train), pre LoRA path resolve")

    save_dir = output_root / train_name
    # z_image_diffsynth (and this job name) save as {train_name}_{steps}.safetensors / {train_name}.safetensors,
    # not *_LoRA_*.safetensors like some SD trainers.
    candidates = list(save_dir.glob("*.safetensors"))
    if not candidates:
        raise RuntimeError(f"No LoRA checkpoint found in {save_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    _log("Z-Image DiffSynth train smoke test ...")
    if not torch.cuda.is_available():
        _log("CUDA not available; skipping train smoke test.")
        return
    device = torch.device("cuda")

    force_regen = os.environ.get("ZIMAGE_TEST_TRAIN_FORCE_REGEN", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )
    image_cache = TEST_TRAIN_IMAGE_CACHE
    _log(f"[DEBUG] image cache dir: {image_cache} (force_regen={force_regen})")

    work_root = Path(tempfile.gettempdir()) / "zimage_diffsynth_train_smoke"
    if work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)
    dataset_dir = work_root / "datasets" / "1"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    prompt = "dog"
    seeds = [42 + i for i in range(NUM_SOURCE_IMAGES)]
    first_seed = seeds[0]

    cache_ok = (not force_regen) and _is_image_cache_valid(image_cache, prompt, seeds)
    if force_regen and image_cache.exists():
        _log("[DEBUG cache] ZIMAGE_TEST_TRAIN_FORCE_REGEN: removing cached images")
        shutil.rmtree(image_cache, ignore_errors=True)

    try:
        if cache_ok:
            _log(
                "1) Skipping source image generation (cache hit — no model/text encoder load for this step)."
            )
            _populate_dataset_from_cache(image_cache, dataset_dir)
            first_before_image = Image.open(image_cache / "000.png").convert("RGB")
            _log(f"   Using cached source images. first_seed={first_seed}")
        else:
            _log("1) Generating source images (cache miss) ...")
            image_cache.mkdir(parents=True, exist_ok=True)
            _cuda_mem_log("before PRE_DATASET build_model")
            base_model = _build_model(device, phase="PRE_DATASET")
            base_pipeline, base_cond, base_uncond = _prepare_generation(base_model, prompt)

            out_paths = [image_cache / f"{idx:03d}.png" for idx in range(NUM_SOURCE_IMAGES)]
            _generate_images_batch(
                base_model,
                base_pipeline,
                base_cond,
                base_uncond,
                prompt,
                seeds,
                out_paths,
                mem_debug_tag="test_train source dataset generate_images",
            )
            for idx in range(NUM_SOURCE_IMAGES):
                (image_cache / f"{idx:03d}.txt").write_text(prompt, encoding="utf-8")
            _write_manifest(image_cache, prompt, seeds)
            _populate_dataset_from_cache(image_cache, dataset_dir)
            first_before_image = Image.open(image_cache / "000.png").convert("RGB")
            _log(f"   Source generation done. first_seed={first_seed}")
            del base_pipeline
            del base_cond
            del base_uncond
            del base_model
            _cuda_mem_log("after delete base_model (post source gen)")
            _release_cuda("after_source_generation")
    except Exception:
        _log("[FATAL] Step 1 (source images / cache) failed:")
        traceback.print_exc()
        raise

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

    marker = work_root / LORA_PATH_MARKER_NAME
    if _env_flag("ZIMAGE_TEST_TRAIN_INPROCESS_TRAIN"):
        try:
            _log("2) Training LoRA in-process (ZIMAGE_TEST_TRAIN_INPROCESS_TRAIN=1) ...")
            _cuda_mem_log("before run_job (train)")
            lora_path = _train_lora(
                work_root=work_root,
                dataset_dir=dataset_dir,
                model_path=model_path,
                sampling_path=sampling_path,
            )
            _log(f"   LoRA checkpoint: {lora_path}")
            _cuda_mem_log("after _train_lora return")
        except Exception:
            _log("[FATAL] Step 2 (training) failed:")
            traceback.print_exc()
            raise
    else:
        _log("2) Training LoRA in a fresh subprocess (parent stays off training VRAM) ...")
        if marker.exists():
            marker.unlink()
        _drop_accelerate_global_singleton("parent before train subprocess")
        _aggressive_cuda_release("parent before train subprocess")
        train_env = os.environ.copy()
        for k in ("ZIMAGE_TEST_TRAIN_TRAIN_SUBPROCESS", "ZIMAGE_TEST_TRAIN_POST_SUBPROCESS"):
            train_env.pop(k, None)
        train_env["ZIMAGE_TEST_TRAIN_TRAIN_SUBPROCESS"] = "1"
        train_env["ZIMAGE_TEST_TRAIN_WORK_ROOT"] = str(work_root)
        train_env["ZIMAGE_TEST_TRAIN_DATASET_DIR"] = str(dataset_dir)
        cmd = [
            sys.executable,
            "-m",
            "extensions_built_in.diffusion_models.z_image_diffsynth.test_train",
        ]
        _log(f"[DEBUG] subprocess train: cwd={TOOLKIT_ROOT}")
        try:
            subprocess.check_call(cmd, env=train_env, cwd=TOOLKIT_ROOT)
        except subprocess.CalledProcessError as exc:
            _log(f"[FATAL] Step 2 train subprocess failed: exit {exc.returncode}")
            raise RuntimeError("train subprocess failed") from exc
        if not marker.is_file():
            raise RuntimeError(f"Train subprocess did not write {marker}")
        lora_path = Path(marker.read_text(encoding="utf-8").strip())
        _log(f"   LoRA checkpoint: {lora_path}")
        _cuda_mem_log("parent after train subprocess (GPU should be idle)")

    after_path = work_root / "after_lora_seed0.png"
    if _env_flag("ZIMAGE_TEST_TRAIN_INPROCESS_POST"):
        _log("3) Post-train generate in-process (ZIMAGE_TEST_TRAIN_INPROCESS_POST=1) ...")
        try:
            _drop_accelerate_global_singleton("before POST_TRAIN")
            _aggressive_cuda_release("immediately before POST_TRAIN build_model")
            _cuda_mem_log("about to call _build_model POST_TRAIN")
            after_model = _build_model(device, phase="POST_TRAIN")
            _cuda_mem_log("after POST_TRAIN build_model")
            _attach_lora_for_inference(after_model, str(lora_path))
            _cuda_mem_log("after attach_lora")
            after_pipeline, after_cond, after_uncond = _prepare_generation(after_model, prompt)
            _generate_images_batch(
                after_model,
                after_pipeline,
                after_cond,
                after_uncond,
                prompt,
                [first_seed],
                [after_path],
                mem_debug_tag="test_train after_lora generate_images",
            )
        except torch.cuda.OutOfMemoryError:
            _log("[FATAL] Step 3: torch.cuda.OutOfMemoryError (VRAM not freed after training).")
            traceback.print_exc()
            _log_cuda_memory_diagnostic("after OOM in step 3")
            raise
        except Exception:
            _log("[FATAL] Step 3 (post-train generation) failed:")
            traceback.print_exc()
            _log_cuda_memory_diagnostic("after exception in step 3")
            raise
    else:
        _log(
            "3) Post-train generate in a fresh subprocess (clean CUDA; see module docstring) ..."
        )
        child_env = os.environ.copy()
        for k in ("ZIMAGE_TEST_TRAIN_TRAIN_SUBPROCESS", "ZIMAGE_TEST_TRAIN_POST_SUBPROCESS"):
            child_env.pop(k, None)
        child_env["ZIMAGE_TEST_TRAIN_POST_SUBPROCESS"] = "1"
        child_env["ZIMAGE_TEST_TRAIN_LORA_PATH"] = str(lora_path)
        child_env["ZIMAGE_TEST_TRAIN_WORK_ROOT"] = str(work_root)
        child_env["ZIMAGE_TEST_TRAIN_FIRST_SEED"] = str(first_seed)
        child_env["ZIMAGE_TEST_TRAIN_PROMPT"] = prompt
        cmd = [
            sys.executable,
            "-m",
            "extensions_built_in.diffusion_models.z_image_diffsynth.test_train",
        ]
        _log(f"[DEBUG] subprocess post-train: cwd={TOOLKIT_ROOT} cmd={cmd!r}")
        try:
            subprocess.check_call(cmd, env=child_env, cwd=TOOLKIT_ROOT)
        except subprocess.CalledProcessError as exc:
            _log(f"[FATAL] Step 3 subprocess failed: exit {exc.returncode}")
            raise RuntimeError("post-train subprocess failed") from exc

    after_image = Image.open(after_path).convert("RGB")

    before_t = _image_to_tensor(first_before_image)
    after_t = _image_to_tensor(after_image)
    cos_sim = _cosine_similarity(before_t, after_t)
    _log(f"4) cosine_similarity(before, after) = {cos_sim:.6f}")

    if math.isnan(cos_sim):
        raise RuntimeError("cosine similarity is NaN")
    if cos_sim >= 0.999999:
        raise RuntimeError(
            f"cosine similarity too high ({cos_sim:.6f}); expected a measurable change after LoRA training"
        )
    _release_cuda("final")

    _log("Done.")


if __name__ == "__main__":
    if _env_flag("ZIMAGE_TEST_TRAIN_TRAIN_SUBPROCESS"):
        _train_subprocess_worker()
    elif _env_flag("ZIMAGE_TEST_TRAIN_POST_SUBPROCESS"):
        _post_train_subprocess_worker()
    else:
        main()

