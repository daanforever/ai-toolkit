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


def _is_quantized_param(param: torch.Tensor) -> bool:
    """True for torchao Float8Tensor / quanto QBytesTensor (and similar)."""
    if hasattr(param, "qdata") and hasattr(param, "scale"):
        return True
    if hasattr(param, "_data") and hasattr(param, "_scale"):
        return True
    return False


def quantized_payload_device(param: torch.Tensor) -> Optional[torch.device]:
    """Device of torchao ``qdata`` / quanto ``_data``, else None."""
    qdata = getattr(param, "qdata", None)
    if qdata is not None:
        return torch.device(qdata.device)
    data = getattr(param, "_data", None)
    if data is not None:
        return torch.device(data.device)
    return None


def safe_module_to_device(
    module: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> None:
    """
    Move module to device (and optionally dtype) without Module.to()/swap_tensors.

    - Quantized weights (torchao/quanto): replace Parameter so payload
      (``qdata``/``_data`` + scale) relocates; ``param.data = ...`` does not.
    - All other params (LoRA, plain bias): identity-preserving ``param.data =``
      so optimizer / PEFT caches keep valid Parameter refs.
    """
    device = torch.device(device)

    for name, param in list(module.named_parameters(recurse=False)):
        need_device = not devices_equal(param.device, device)
        payload = quantized_payload_device(param)
        if payload is not None and not devices_equal(payload, device):
            need_device = True
        need_dtype = dtype is not None and param.dtype != dtype
        if not need_device and not need_dtype:
            continue
        if dtype is not None:
            moved = param.to(device=device, dtype=dtype)
        else:
            moved = param.to(device=device)
        if _is_quantized_param(param):
            module._parameters[name] = nn.Parameter(
                moved, requires_grad=param.requires_grad
            )
        else:
            param.data = moved.data if hasattr(moved, "data") else moved

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
