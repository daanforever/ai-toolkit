# LoRA target modules and weight conversion for Z-Image DiffSynth (DiffSynth-Studio convention).
# Attention: to_q, to_k, to_v, to_out.0; FeedForward: w1, w2, w3 (see diffsynth/models/z_image_dit.py).
# Load convention: prefix "diffusion_model" (diffsynth/utils/lora/general.py).

from typing import Dict, Any

# Class names that contain the linear layers we want for LoRA (toolkit matches by __class__.__name__).
TARGET_LORA_MODULES = ["Attention", "FeedForward"]


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
