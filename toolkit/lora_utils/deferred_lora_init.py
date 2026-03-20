"""Finalize LoRA init after checkpoint load (deferred PiSSA for weights not in file)."""

from __future__ import annotations

from typing import AbstractSet


def lora_down_present_in_loaded_keys(lora_name: str, loaded_keys: AbstractSet[str]) -> bool:
    """
    True if ``loaded_keys`` contains any tensor key for this LoRA module's lora_down.

    Uses the first path segment (same as ``add_module(lora_name, ...)`` / ``state_dict``)
    so we do not mis-detect via naive ``startswith`` (e.g. ``foo`` vs ``foo_bar``).
    """
    if f"{lora_name}.lora_down.weight" in loaded_keys:
        return True
    for k in loaded_keys:
        if "lora_down" not in k:
            continue
        head, _, _ = k.partition(".")
        if head == lora_name:
            return True
    return False


def finalize_deferred_lora_init(network, loaded_keys: AbstractSet[str]) -> None:
    if not getattr(network, "deferred_lora_init", False):
        return
    for lora in network.text_encoder_loras + network.unet_loras:
        fin = getattr(lora, "finalize_deferred_lora_init_if_needed", None)
        if callable(fin):
            fin(loaded_keys)
