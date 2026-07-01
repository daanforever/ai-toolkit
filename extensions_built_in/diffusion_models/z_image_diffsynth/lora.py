# LoRA target modules and weight conversion for Z-Image DiffSynth (DiffSynth-Studio convention).
# Attention: to_q, to_k, to_v, to_out.0; FeedForward: w1, w2, w3 (see diffsynth/models/z_image_dit.py).
# Load convention: prefix "diffusion_model" (diffsynth/utils/lora/general.py).

import re
from collections import OrderedDict
from typing import Dict, Any, List, Optional

# Class names that contain the linear layers we want for LoRA (toolkit matches by __class__.__name__).
TARGET_LORA_MODULES = ["Attention", "FeedForward"]

_DIT_BLOCK_KEY_PATTERNS = (
    re.compile(
        r"^(?:transformer|lora_transformer)\$\$_inner_dit\$\$"
        r"(layers|noise_refiner|context_refiner)\$\$(\d+)(?:\$\$|$)"
    ),
    re.compile(
        r"^(?:transformer|lora_transformer)\._inner_dit\."
        r"(layers|noise_refiner|context_refiner)\.(\d+)(?:\.|$)"
    ),
    re.compile(
        r"^(?:lora_unet|lora_transformer)__inner_dit_"
        r"(layers|noise_refiner|context_refiner)_(\d+)(?:_|$)"
    ),
)


def parse_lora_block_key(lora_name: str) -> Optional[str]:
    """Parse zimage DiT block key from LoRA module name.

    Supported formats:
    - transformer$$_inner_dit$$layers$$0$$attention$$to_q
    - lora_unet__inner_dit_layers_0_attention_to_q
    - lora_transformer__inner_dit_layers_0_attention_to_q
    """
    for pattern in _DIT_BLOCK_KEY_PATTERNS:
        match = pattern.match(lora_name)
        if match is None:
            continue
        block_name, block_index = match.groups()
        return f"{block_name}_{int(block_index)}"
    return None


def _block_sort_key(block_key: str):
    if block_key == "other":
        return (99, 0, block_key)
    if "_" not in block_key:
        return (98, 0, block_key)
    prefix, suffix = block_key.rsplit("_", 1)
    if not suffix.isdigit():
        return (98, 0, block_key)
    prefix_order = {
        "layers": 0,
        "noise_refiner": 1,
        "context_refiner": 2,
    }.get(prefix, 50)
    return (prefix_order, int(suffix), block_key)


def group_loras_by_block(loras: List[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for lora in loras:
        block_key = parse_lora_block_key(lora.lora_name)
        if block_key is None:
            block_key = "other"
            print(
                f"[zimage_diffsynth] unknown DiT LoRA name for param grouping: {lora.lora_name}"
            )
        grouped.setdefault(block_key, []).append(lora)

    ordered = OrderedDict()
    for block_key in sorted(grouped.keys(), key=_block_sort_key):
        ordered[block_key] = grouped[block_key]
    return ordered


def convert_lora_weights_before_save(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert toolkit LoRA keys to DiffSynth convention for saving.
    Toolkit uses prefix 'transformer' or 'lora_transformer'; DiffSynth expects 'diffusion_model'.
    """
    new_sd = {}
    for key, value in state_dict.items():
        new_key = key.replace("transformer.", "diffusion_model.").replace(
            "lora_transformer.", "diffusion_model."
        )
        new_sd[new_key] = value
    return new_sd


def convert_lora_weights_before_load(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert DiffSynth LoRA keys to toolkit convention for loading.
    DiffSynth uses prefix 'diffusion_model'; toolkit expects 'transformer' / 'lora_transformer'.
    """
    new_sd = {}
    for key, value in state_dict.items():
        new_key = key.replace("diffusion_model.", "transformer.")
        new_sd[new_key] = value
    return new_sd


def convert_accuracy_recovery_weights_before_load(
    state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert ARA state dict keys to the format expected by LoRASpecialNetwork.load_weights
    when applied to the raw DiT (prefix lora_transformer_, underscores, lora_down/lora_up).
    Input keys are assumed to be already in 'transformer.xxx' form (e.g. after
    convert_lora_weights_before_load). Same convention as z_image for accuracy recovery.
    """
    new_sd = {}
    for key, value in state_dict.items():
        if ".lora_A." in key or ".lora_B." in key:
            parts = key.split(".")
            # e.g. transformer.layers.0.attention.to_q.lora_A.weight -> lora_transformer_layers_0_attention_to_q.lora_down.weight
            module_path = "lora_transformer_" + "_".join(parts[1:-2])
            param = "lora_down" if "lora_A" in key else "lora_up"
            new_key = f"{module_path}.{param}.{parts[-1]}"
            new_sd[new_key] = value
        else:
            new_sd[key] = value
    return new_sd
