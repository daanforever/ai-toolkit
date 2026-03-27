"""
LoRA loading and application for Z-Image pipeline.
Supports two modes: set_adapters (dynamic) or fuse_lora (merge into base model).
"""

import glob
import gc
import os
import re

from diffusers.loaders.lora_pipeline import ZImageLoraLoaderMixin
from safetensors.torch import load_file as load_safetensors_file


def parse_loras_config(loras_raw):
    """
    Parse loras from config. Supports:
    - list of specs -> adapters list, fuse_lora default True
    - dict with fuse_lora and adapters -> (adapters list, fuse_lora bool)
    Returns (adapters_list, fuse_lora).
    """
    if loras_raw is None:
        return [], True
    if isinstance(loras_raw, list):
        return loras_raw, True
    d = loras_raw if isinstance(loras_raw, dict) else {}
    adapters = d.get("adapters") or d.get("items") or []
    if not isinstance(adapters, list):
        adapters = []
    fuse_lora = d.get("fuse_lora", True)
    return adapters, fuse_lora


def normalize_path(p: str) -> str:
    if not p:
        return p
    return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))


def normalize_zimage_lora_keys_for_diffusers(state_dict: dict) -> dict:
    """
    Map toolkit / DiffSynth checkpoint key layouts to Diffusers Z-Image LoRA conversion
    (diffusion_model.*.lora_down.weight / .lora_up.weight / .alpha). All logic stays in this
    script only — no changes to the training codebase.
    """
    out = {}
    for k, v in state_dict.items():
        nk = k
        while nk.startswith("."):
            nk = nk[1:]
        if "inner.dit." in nk:
            nk = nk.replace("inner.dit.", "diffusion_model.", 1)
        if nk.startswith("lora_transformer__inner_dit_"):
            nk = "diffusion_model." + nk[len("lora_transformer__inner_dit_") :]
        elif nk.startswith("lora_unet._inner_dit."):
            nk = "diffusion_model." + nk[len("lora_unet._inner_dit.") :]
        nk = re.sub(r"attention\.to\.([qkv])\.lora", r"attention.to_\1.lora", nk)
        nk = re.sub(r"attention\.to\.([qkv])(\.|$)", r"attention.to_\1\2", nk)
        nk = nk.replace(".lora.down.weight", ".lora_down.weight")
        nk = nk.replace(".lora.up.weight", ".lora_up.weight")
        nk = nk.replace(".lora.down.", ".lora_down.")
        nk = nk.replace(".lora.up.", ".lora_up.")
        out[nk] = v
    return out


def filter_zimage_lora_keys_for_diffusers(state_dict: dict) -> dict:
    """Drop tensors Diffusers does not map (e.g. lora_up.bias)."""
    out = {}
    for k, v in state_dict.items():
        if k.endswith(".lora_up.bias") or k.endswith(".lora_down.bias"):
            continue
        out[k] = v
    return out


def prepare_zimage_lora_state_dict_for_diffusers(state_dict: dict) -> dict:
    return filter_zimage_lora_keys_for_diffusers(
        normalize_zimage_lora_keys_for_diffusers(state_dict)
    )


def _load_local_lora_safetensors_dict(path: str, weight_name: str | None) -> dict:
    """Load a single .safetensors LoRA from a file path or a local directory."""
    path = normalize_path(path)
    if os.path.isfile(path) and path.lower().endswith(".safetensors"):
        return dict(load_safetensors_file(path))
    if os.path.isdir(path):
        if weight_name:
            fp = os.path.join(path, weight_name)
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"LoRA weight not found: {fp}")
            return dict(load_safetensors_file(fp))
        from diffusers.loaders.lora_base import LORA_WEIGHT_NAME_SAFE

        candidate = os.path.join(path, LORA_WEIGHT_NAME_SAFE)
        if os.path.isfile(candidate):
            return dict(load_safetensors_file(candidate))
        matches = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if len(matches) == 1:
            return dict(load_safetensors_file(matches[0]))
        raise FileNotFoundError(
            f"Could not pick a LoRA .safetensors in {path} "
            f"(set weight_name or use a single .safetensors file)."
        )
    raise ValueError(f"Expected a local .safetensors file or directory, got: {path}")


def validate_loras(adapters_list, normalize_path_func=None):
    """
    Validate adapter specs and paths. normalize_path_func(path) -> path.
    Returns list of normalized specs (with path normalized). Exits on error.
    """
    norm = normalize_path_func or normalize_path
    out = []
    for i, spec in enumerate(adapters_list):
        path = spec.get("path") or spec.get("file")
        if not path:
            raise SystemExit(f"LoRA entry {i + 1}: missing 'path' or 'file'")
        path = norm(path)
        if os.path.isfile(path) and not path.lower().endswith(".safetensors"):
            raise SystemExit(f"LoRA file must be .safetensors: {path}")
        if not os.path.isfile(path) and not os.path.isdir(path):
            raise SystemExit(f"LoRA path not found: {path}")
        out.append({**spec, "path": path})
    return out


def load_loras_into_transformer(
    transformer,
    adapters_list,
    normalize_path_func=None,
    debug=False,
):
    """
    Load LoRA state dicts into transformer (for local pipeline build when fuse_lora is False).
    """
    norm = normalize_path_func or normalize_path
    for i, spec in enumerate(adapters_list):
        path = spec.get("path") or spec.get("file")
        path = norm(path)
        adapter_name = f"lora_{i}"
        if debug:
            print(f"  [debug] Before loading LoRA {i}...")
        raw = _load_local_lora_safetensors_dict(path, spec.get("weight_name"))
        raw = prepare_zimage_lora_state_dict_for_diffusers(raw)
        state_dict, metadata = ZImageLoraLoaderMixin.lora_state_dict(
            raw, return_lora_metadata=True
        )
        ZImageLoraLoaderMixin.load_lora_into_transformer(
            state_dict,
            transformer=transformer,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=None,
        )
        del state_dict, metadata
        gc.collect()
        if debug:
            print(f"  [debug] After loading LoRA {i}.")


def apply_loras_to_pipeline(
    pipe,
    adapters_list,
    *,
    fuse_lora=True,
    normalize_path_func=None,
    debug=False,
):
    """
    Apply LoRAs to an assembled ZImagePipeline.

    - If fuse_lora is True: for each adapter, load_lora_weights -> fuse_lora(lora_scale=weight) -> unload_lora_weights.
    - If fuse_lora is False: load_lora_weights for each, then set_adapters(adapter_names, adapter_weights).
    """
    if not adapters_list:
        return
    norm = normalize_path_func or normalize_path
    if fuse_lora:
        for i, spec in enumerate(adapters_list):
            path = spec.get("path") or spec.get("file")
            path = norm(path)
            weight = float(spec.get("weight", 1.0))
            adapter_name = f"lora_{i}"
            raw = _load_local_lora_safetensors_dict(path, spec.get("weight_name"))
            raw = prepare_zimage_lora_state_dict_for_diffusers(raw)
            pipe.load_lora_weights(raw, adapter_name=adapter_name)
            pipe.set_adapters([adapter_name], adapter_weights=[weight])
            pipe.fuse_lora(lora_scale=weight, components=["transformer"])
            pipe.unload_lora_weights()
            if debug:
                print(f"  [debug] Fused LoRA {i} (weight={weight}).")
    else:
        print("Loading LoRA(s)...")
        for i, spec in enumerate(adapters_list):
            path = spec.get("path") or spec.get("file")
            path = norm(path)
            adapter_name = f"lora_{i}"
            raw = _load_local_lora_safetensors_dict(path, spec.get("weight_name"))
            raw = prepare_zimage_lora_state_dict_for_diffusers(raw)
            pipe.load_lora_weights(raw, adapter_name=adapter_name)
        adapter_names = [f"lora_{i}" for i in range(len(adapters_list))]
        adapter_weights = [float(spec.get("weight", 1.0)) for spec in adapters_list]
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
