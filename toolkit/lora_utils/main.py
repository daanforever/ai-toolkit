from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from toolkit.lora_special import LoRASpecialNetwork


def share_module_params_and_buffers(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """
    Make all parameters and buffers of ``target`` reference those of ``source``.
    """
    # share all parameters
    for name, param in source.named_parameters():
        parts = name.split(".")
        obj = target
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], param)

    # and all registered buffers (e.g. alpha)
    for name, buf in source.named_buffers():
        parts = name.split(".")
        obj = target
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], buf)


def share_lora_pair(my_lora: torch.nn.Module, other_lora: torch.nn.Module) -> None:
    """
    Share parameters and buffers between two LoRA modules with basic safety checks.
    """
    assert type(my_lora) is type(other_lora), "lora module type mismatch"
    assert getattr(my_lora, "lora_name", None) == getattr(other_lora, "lora_name", None), (
        f"lora name mismatch: {getattr(my_lora, 'lora_name', None)} vs {getattr(other_lora, 'lora_name', None)}"
    )
    share_module_params_and_buffers(my_lora, other_lora)


def share_network_parameters(target_net: LoRASpecialNetwork, source_net: LoRASpecialNetwork) -> None:
    """
    Share all trainable parameters and buffers between two ``LoRASpecialNetwork`` instances
    of the same structure.
    """
    assert len(target_net.unet_loras) == len(source_net.unet_loras), "unet_loras length mismatch"
    assert len(target_net.text_encoder_loras) == len(source_net.text_encoder_loras), "text_encoder_loras length mismatch"

    for my_lora, other_lora in zip(target_net.unet_loras, source_net.unet_loras):
        share_lora_pair(my_lora, other_lora)
    for my_lora, other_lora in zip(target_net.text_encoder_loras, source_net.text_encoder_loras):
        share_lora_pair(my_lora, other_lora)

    # If we are also retraining main in/out layers, make sure those parameters are shared too.
    assert target_net.full_train_in_out == source_net.full_train_in_out, "full_train_in_out flag mismatch"
    if target_net.full_train_in_out:
        for attr_name in ("unet_conv_in", "unet_conv_out", "transformer_pos_embed", "transformer_proj_out"):
            my_module = getattr(target_net, attr_name, None)
            other_module = getattr(source_net, attr_name, None)
            if my_module is None or other_module is None:
                continue
            assert type(my_module) is type(other_module), f"{attr_name} module type mismatch"
            share_module_params_and_buffers(my_module, other_module)

