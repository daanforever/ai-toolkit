# Load Z-Image components using main-project paths (z_image style) and DiffSynth DiT/VAE interface.
# DiffSynth-Studio must be present in z_image_diffsynth/DiffSynth-Studio (git clone).

import os
import sys
import glob
from typing import Optional, Tuple, List, Any

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM
from diffusers import AutoencoderKL

from toolkit.paths import normalize_path
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
    quantize_transformer: bool = False,
    base_model: Optional[Any] = None,
    sampling_loader_mode: str = "auto",
) -> dict:
    """
    Load tokenizer, text_encoder, vae, dit (and optionally sampling dit) from paths.
    Paths resolved like z_image: model_path, base_model_path (extras_name_or_path), transformer in model_path/transformer.
    Returns dict with: tokenizer, text_encoder, vae, vae_encoder, vae_decoder, dit, sampling_dit (optional).
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

    # 1) Sampling transformer first when configured (VRAM control)
    # When sampling_name_or_path points to the same diffusers-style checkpoint as z_image uses,
    # we can load it as ZImageTransformer2DModel and run it with ZImagePipeline (same as z_image),
    # or fall back to DiffSynth ZImageDiT and model_fn_z_image_turbo. The behaviour is controlled
    # by sampling_loader_mode: \"auto\" (default), \"diffusers\", or \"diffsynth\".
    sampling_dit = None
    sampling_is_diffusers = False
    mode = (sampling_loader_mode or "auto").lower()
    if mode not in ("auto", "diffusers", "diffsynth"):
        log(f"Unknown sampling_loader_mode='{sampling_loader_mode}', falling back to 'auto'")
        mode = "auto"

    if sampling_transformer_path:
        sampling_transformer_path = normalize_path(sampling_transformer_path)
        sp_transformer_folder = os.path.join(sampling_transformer_path, "transformer")
        if not os.path.isdir(sp_transformer_folder):
            sp_transformer_folder = sampling_transformer_path

        # diffusers path (ZImageTransformer2DModel) when allowed by mode
        if mode in ("auto", "diffusers"):
            try:
                from extensions_built_in.diffusion_models.z_image.loading import (
                    load_zimage_transformer_from_shards,
                )
                log("Loading sampling transformer (diffusers ZImage format)")
                sampling_dit = load_zimage_transformer_from_shards(
                    sp_transformer_folder,
                    subfolder=None,
                    torch_dtype=dtype,
                    device=device,
                )
                sampling_is_diffusers = True
            except (ValueError, FileNotFoundError, OSError, RuntimeError) as e:
                if mode == "diffusers":
                    # Explicit diffusers-only mode: surface a clear error instead of
                    # silently falling back to DiffSynth.
                    raise RuntimeError(
                        f"Failed to load sampling transformer in 'diffusers' mode from '{sp_transformer_folder}': {e}"
                    ) from e
                # auto mode: fall back to DiffSynth DiT below.

        # DiffSynth DiT path when requested explicitly or diffusers failed / was skipped
        if sampling_dit is None and mode in ("auto", "diffsynth"):
            log("Loading sampling transformer (DiT)")
            sampling_dit = load_dit_from_folder(sp_transformer_folder, dtype, device)

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
        sampling_dit.to("cpu")
        flush()

    # 2) Main DiT
    transformer_folder = os.path.join(model_path, "transformer")
    if not os.path.isdir(transformer_folder):
        transformer_folder = model_path
    log("Loading transformer (DiT)")
    dit = load_dit_from_folder(transformer_folder, dtype, device)
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
    dit.to("cpu")
    flush()

    # 3) Tokenizer & text encoder (same as z_image)
    log("Loading tokenizer and text encoder")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        base_model_path, subfolder="text_encoder", dtype=dtype
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
        "sampling_dit": sampling_dit,
        "sampling_is_diffusers": sampling_is_diffusers,
    }
