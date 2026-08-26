# Load Z-Image components using main-project paths (z_image style) and DiffSynth DiT/VAE interface.
# DiffSynth-Studio must be present in z_image_diffsynth/DiffSynth-Studio (git clone).

import os
import sys
import glob
from typing import Optional, Tuple, List, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3ForCausalLM
from diffusers import AutoencoderKL

from toolkit.paths import normalize_path
from toolkit.util.device import safe_module_to_device
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.util.debug import is_debug_enabled
from toolkit.basic import flush
from optimum.quanto import freeze

from .vae_wrapper import DiffSynthVAEWrapper

_DIFFSYNTH_IMPORTED = False


def _ensure_diffsynth_path():
    """Add DiffSynth-Studio to sys.path so 'diffsynth' can be imported."""
    global _DIFFSYNTH_IMPORTED
    if _DIFFSYNTH_IMPORTED:
        return
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ds_dir = os.path.join(this_dir, "DiffSynth-Studio")
    if not os.path.isdir(ds_dir):
        raise ImportError(
            "Установите DiffSynth-Studio: клонируйте репозиторий в "
            "z_image_diffsynth/DiffSynth-Studio или выполните pip install -e DiffSynth-Studio."
        )
    if ds_dir not in sys.path:
        sys.path.insert(0, ds_dir)
    _DIFFSYNTH_IMPORTED = True


def _load_state_dict_from_folder(folder: str, dtype=None, device="cpu") -> dict:
    """Load and merge state dict from a folder of .safetensors files."""
    from safetensors.torch import load_file
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {folder}")
    state_dict = {}
    for f in files:
        shard = load_file(f, device="cpu")
        if dtype is not None:
            for k in list(shard.keys()):
                if shard[k].dtype != dtype:
                    shard[k] = shard[k].to(dtype)
        state_dict.update(shard)
        del shard
    return state_dict


def load_dit_from_folder(
    transformer_path: str,
    dtype: torch.dtype,
    device: torch.device,
    config: Optional[dict] = None,
) -> Any:
    """Load DiffSynth ZImageDiT from a folder of safetensors (DiffSynth-format checkpoint)."""
    _ensure_diffsynth_path()
    from diffsynth.models.z_image_dit import ZImageDiT
    import gc

    config = config or {}
    state_dict = _load_state_dict_from_folder(transformer_path, dtype=dtype, device="cpu")
    # Optional: strip prefix if present (e.g. transformer. or pipe.dit.)
    for prefix in ("transformer.", "pipe.dit."):
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            break
    with torch.device("meta"):
        model = ZImageDiT(**config)
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    if len(missing) > 100:
        raise RuntimeError(
            "DiT checkpoint does not match ZImageDiT (too many missing keys). "
            "Use a DiffSynth-format Z-Image checkpoint (e.g. from DiffSynth-Studio examples)."
        )
    
    # Clean up state dict dictionary before GPU transfer to release CPU RAM
    del state_dict
    gc.collect()

    model = model.to(dtype=dtype, device=device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _normalize_loader_mode(loader_mode: Optional[str], log) -> str:
    mode = (loader_mode or "auto").lower()
    if mode not in ("auto", "diffusers", "diffsynth"):
        log(f"Unknown loader_mode='{loader_mode}', falling back to 'auto'")
        return "auto"
    return mode


def _load_transformer_by_mode(
    transformer_folder: str,
    dtype: torch.dtype,
    device: torch.device,
    mode: str,
    log,
    label: str,
) -> Tuple[Any, bool]:
    """
    Load a Z-Image transformer as Diffusers ZImageTransformer2DModel or DiffSynth ZImageDiT.
    mode: \"auto\" | \"diffusers\" | \"diffsynth\".
    Returns (module, is_diffusers).
    """
    dit = None
    is_diffusers = False

    if mode in ("auto", "diffusers"):
        try:
            from extensions_built_in.diffusion_models.z_image.loading import (
                load_zimage_transformer_from_shards,
            )
            log(f"Loading {label} (diffusers ZImage format)")
            dit = load_zimage_transformer_from_shards(
                transformer_folder,
                subfolder=None,
                torch_dtype=dtype,
                device=device,
            )
            is_diffusers = True
        except (ValueError, FileNotFoundError, OSError, RuntimeError) as e:
            if mode == "diffusers":
                raise RuntimeError(
                    f"Failed to load {label} in 'diffusers' mode from '{transformer_folder}': {e}"
                ) from e
            # auto: fall back to DiffSynth DiT below.

    if dit is None and mode in ("auto", "diffsynth"):
        log(f"Loading {label} (DiffSynth DiT)")
        dit = load_dit_from_folder(transformer_folder, dtype, device)

    return dit, is_diffusers


def load_components(
    model_path: str,
    base_model_path: Optional[str],
    *,
    dtype: torch.dtype,
    device: torch.device,
    log_fn=None,
    quantize_te: bool = False,
    qtype_te: str = "float8",
    sampling_transformer_path: Optional[str] = None,
    te_name_or_path: Optional[str] = None,
    quantize_transformer: bool = False,
    base_model: Optional[Any] = None,
    loader_mode: str = "auto",
) -> dict:
    """
    Load tokenizer, text_encoder, vae, dit (and optionally sampling dit) from paths.
    Paths resolved like z_image: model_path, base_model_path (extras_name_or_path), transformer in model_path/transformer.
    te_name_or_path: when set, load tokenizer + TE from that HF id / local root without
    subfolders (standalone CausalLM repos). When falsy, use base_model_path with
    subfolder=\"tokenizer\" / \"text_encoder\" and Qwen3ForCausalLM (Z-Image snapshot layout).
    VAE always from base_model_path. loader_mode applies to both transformers.
    Returns dict with: tokenizer, text_encoder, vae, vae_encoder, vae_decoder, dit,
    dit_is_diffusers, sampling_dit (optional), sampling_is_diffusers.
    """
    _ensure_diffsynth_path()
    model_path = normalize_path(model_path)
    if base_model_path is None:
        base_model_path = normalize_path(model_path)
    else:
        base_model_path = normalize_path(base_model_path)

    def log(msg):
        if log_fn:
            log_fn(msg)

    mode = _normalize_loader_mode(loader_mode, log)

    # 1) Sampling transformer first when configured (VRAM control)
    sampling_dit = None
    sampling_is_diffusers = False

    if sampling_transformer_path:
        sampling_transformer_path = normalize_path(sampling_transformer_path)
        sp_transformer_folder = os.path.join(sampling_transformer_path, "transformer")
        if not os.path.isdir(sp_transformer_folder):
            sp_transformer_folder = sampling_transformer_path

        sampling_dit, sampling_is_diffusers = _load_transformer_by_mode(
            sp_transformer_folder, dtype, device, mode, log, "sampling transformer"
        )

        if quantize_transformer and base_model is not None:
            log("Quantizing sampling transformer")
            if is_debug_enabled():
                ara = getattr(
                    base_model.model_config, "accuracy_recovery_adapter", None
                )
                if ara:
                    log(
                        f"[z_image_diffsynth ARA] Applying accuracy recovery adapter to sampling transformer: {ara}"
                    )
                else:
                    log(
                        "[z_image_diffsynth ARA] Quantizing sampling transformer without ARA (accuracy_recovery_adapter not set)"
                    )
            quantize_model(base_model, sampling_dit)
            flush()
        safe_module_to_device(sampling_dit, torch.device("cpu"))
        flush()

    # 2) Main transformer (Diffusers or DiffSynth per loader_mode)
    transformer_folder = os.path.join(model_path, "transformer")
    if not os.path.isdir(transformer_folder):
        transformer_folder = model_path
    dit, dit_is_diffusers = _load_transformer_by_mode(
        transformer_folder, dtype, device, mode, log, "transformer"
    )
    if quantize_transformer and base_model is not None:
        log("Quantizing transformer")
        if is_debug_enabled():
            ara = getattr(
                base_model.model_config, "accuracy_recovery_adapter", None
            )
            if ara:
                log(
                    f"[z_image_diffsynth ARA] Applying accuracy recovery adapter to main transformer: {ara}"
                )
            else:
                log(
                    "[z_image_diffsynth ARA] Quantizing main transformer without ARA (accuracy_recovery_adapter not set)"
                )
        quantize_model(base_model, dit)
        flush()
    # Move main DiT to CPU after (optional) quantization to reduce VRAM
    # usage during setup; training/device presets and call-sites will move
    # it back to the appropriate device when actually used.
    safe_module_to_device(dit, torch.device("cpu"))
    flush()

    # 3) Tokenizer & text encoder (Lumina2-style override vs Z-Image snapshot)
    log("Loading tokenizer and text encoder")
    if te_name_or_path:
        te_root = normalize_path(te_name_or_path)
        log(f"Loading standalone TE from {te_root} (no subfolder)")
        tokenizer = AutoTokenizer.from_pretrained(te_root)
        text_encoder = AutoModelForCausalLM.from_pretrained(te_root, dtype=dtype)
    else:
        te_root = base_model_path
        tokenizer = AutoTokenizer.from_pretrained(te_root, subfolder="tokenizer")
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            te_root, subfolder="text_encoder", dtype=dtype
        )
    text_encoder.to(device)
    if quantize_te:
        qtype = get_qtype(qtype_te or "float8")
        qtype_name = getattr(qtype, "name", str(qtype))
        log(f"Quantizing text encoder (weights={qtype_name})")
        quantize(text_encoder, weights=qtype)
        freeze(text_encoder)
        flush()
        log("Text encoder quantized")

    # 4) VAE (single AutoencoderKL, wrap as encoder+decoder interface)
    log("Loading VAE")
    vae = AutoencoderKL.from_pretrained(
        base_model_path, subfolder="vae", torch_dtype=dtype
    )
    vae_wrapper = DiffSynthVAEWrapper(vae, None)  # single VAE: encode/decode via vae

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "vae_encoder": vae,
        "vae_decoder": vae,
        "vae_wrapper": vae_wrapper,
        "dit": dit,
        "dit_is_diffusers": dit_is_diffusers,
        "sampling_dit": sampling_dit,
        "sampling_is_diffusers": sampling_is_diffusers,
    }
