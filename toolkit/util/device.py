"""
Device utilities. safe_module_to_device moves a module without Module.to(),
avoiding PyTorch's swap_tensors path which fails on quantized parameters
(e.g. QLinear) that have weakrefs or requires_grad=False.
"""
from typing import Optional

import torch
import torch.nn as nn


def devices_equal(a: torch.device, b: torch.device) -> bool:
    """Compare devices treating cuda vs cuda:0 as the same current device."""
    if a.type != b.type:
        return False
    if a.type == "cpu":
        return True
    a_idx = a.index if a.index is not None else torch.cuda.current_device()
    b_idx = b.index if b.index is not None else torch.cuda.current_device()
    return a_idx == b_idx


def safe_module_to_device(
    module: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> None:
    """
    Move module to device (and optionally dtype) by replacing registered
    parameters/buffers. Avoids Module.to() / swap_tensors.

    Replacing via ``nn.Parameter(tensor.to(device))`` correctly relocates
    quanto QBytesTensor ``_data`` / ``_scale``; assigning ``param.data = ...``
    does not.
    """
    device = torch.device(device)

    for name, param in list(module.named_parameters(recurse=False)):
        need_device = not devices_equal(param.device, device)
        need_dtype = dtype is not None and param.dtype != dtype
        if not need_device and not need_dtype:
            continue
        if dtype is not None:
            moved = param.to(device=device, dtype=dtype)
        else:
            moved = param.to(device=device)
        module._parameters[name] = nn.Parameter(
            moved, requires_grad=param.requires_grad
        )

    for name, buf in list(module.named_buffers(recurse=False)):
        if buf is None:
            continue
        need_device = not devices_equal(buf.device, device)
        need_dtype = dtype is not None and buf.dtype != dtype
        if not need_device and not need_dtype:
            continue
        if dtype is not None:
            module._buffers[name] = buf.to(device=device, dtype=dtype)
        else:
            module._buffers[name] = buf.to(device=device)

    for _, child in module.named_children():
        safe_module_to_device(child, device, dtype)
