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


_PACK_ALIGNMENT = 512  # bytes; same alignment the CUDA caching allocator gives each block


def _pack_frozen_params(
    module: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype],
) -> None:
    """Move the subtree's frozen plain params into one allocation on ``device`` (as views)."""
    params = [
        p
        for p in module.parameters()
        if not p.requires_grad
        and not _is_quantized_param(p)
        and p.is_contiguous()
        and (not devices_equal(p.device, device) or (dtype is not None and p.dtype != dtype))
    ]
    if not params:
        return
    offsets = []
    total = 0
    for p in params:
        total = -(-total // _PACK_ALIGNMENT) * _PACK_ALIGNMENT
        offsets.append(total)
        total += p.numel() * (dtype or p.dtype).itemsize
    storage = torch.empty(total, dtype=torch.uint8, device=device)
    for p, offset in zip(params, offsets):
        target_dtype = dtype or p.dtype
        packed = storage[offset:offset + p.numel() * target_dtype.itemsize]
        packed = packed.view(target_dtype).view(p.shape)
        packed.copy_(p.data)
        p.data = packed


def safe_module_to_device(
    module: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    pack: bool = False,
) -> None:
    """
    Move module to device (and optionally dtype) without Module.to()/swap_tensors.

    - Quantized weights (torchao/quanto): replace Parameter so payload
      (``qdata``/``_data`` + scale) relocates; ``param.data = ...`` does not.
    - All other params (LoRA, plain bias): identity-preserving ``param.data =``
      so optimizer / PEFT caches keep valid Parameter refs.
    - ``pack``: frozen plain params of the whole subtree share one allocation, so the
      CUDA caching allocator does not round every large weight up to its own 2 MiB
      segment. The allocation is freed only once every one of those params has moved
      off ``device``; moving a single submodule away keeps it alive.
    """
    device = torch.device(device)
    if pack:
        _pack_frozen_params(module, device, dtype)

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
        if _is_quantized_param(param) and (
            quantized_payload_device(moved) is not None
            or hasattr(moved, "qdata")
            or hasattr(moved, "_data")
        ):
            # Real torchao/quanto: .to() relocates payload; replace Parameter.
            module._parameters[name] = nn.Parameter(
                moved, requires_grad=param.requires_grad
            )
        else:
            # Plain Parameter (incl. bolted-on qdata fixtures): keep identity.
            param.data = moved.data if hasattr(moved, "data") else moved
            for attr in ("qdata", "_data", "scale", "_scale"):
                val = getattr(param, attr, None)
                if isinstance(val, torch.Tensor) and not devices_equal(val.device, device):
                    setattr(param, attr, val.to(device))

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
