"""
LoRA loading and application for Z-Image pipeline.
Supports two modes: set_adapters (dynamic) or fuse_lora (merge into base model).
"""

import gc
import os

from diffusers.loaders.lora_pipeline import ZImageLoraLoaderMixin


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
        if os.path.isfile(path) and path.lower().endswith(".safetensors"):
            lora_dir = os.path.dirname(path) or "."
            weight_name = os.path.basename(path)
            state_dict, metadata = ZImageLoraLoaderMixin.lora_state_dict(
                lora_dir, weight_name=weight_name, return_lora_metadata=True
            )
        else:
            state_dict, metadata = ZImageLoraLoaderMixin.lora_state_dict(
                path, return_lora_metadata=True
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
            if os.path.isfile(path) and path.lower().endswith(".safetensors"):
                lora_dir = os.path.dirname(path) or "."
                weight_name = os.path.basename(path)
                pipe.load_lora_weights(
                    lora_dir, weight_name=weight_name, adapter_name=adapter_name
                )
            else:
                pipe.load_lora_weights(path, adapter_name=adapter_name)
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
            if os.path.isfile(path) and path.lower().endswith(".safetensors"):
                lora_dir = os.path.dirname(path) or "."
                weight_name = os.path.basename(path)
                pipe.load_lora_weights(
                    lora_dir, weight_name=weight_name, adapter_name=adapter_name
                )
            else:
                pipe.load_lora_weights(path, adapter_name=adapter_name)
        adapter_names = [f"lora_{i}" for i in range(len(adapters_list))]
        adapter_weights = [float(spec.get("weight", 1.0)) for spec in adapters_list]
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
