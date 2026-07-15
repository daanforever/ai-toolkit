"""Ensure LoRA down weights remain nonzero after cast to bfloat16."""

from __future__ import annotations

import torch


def ensure_bf16_nonzero_(tensor: torch.Tensor) -> int:
    """In-place replace values that become exact 0 in bf16 with ±smallest_normal.

    Sign is taken from the current value; for exact zeros a random ±1 is used.
    Returns the number of elements replaced.
    """
    with torch.no_grad():
        mask = tensor.to(torch.bfloat16) == 0
        n = int(mask.sum().item())
        if n == 0:
            return 0
        eps = torch.finfo(torch.bfloat16).smallest_normal
        selected = tensor[mask]
        signs = torch.sign(selected)
        zero_sign = signs == 0
        if bool(zero_sign.any()):
            rnd = torch.rand(int(zero_sign.sum().item()), device=tensor.device)
            replacement = torch.where(
                rnd > 0.5,
                torch.ones_like(rnd),
                -torch.ones_like(rnd),
            )
            signs = signs.clone()
            signs[zero_sign] = replacement.to(dtype=signs.dtype)
        tensor[mask] = (signs * eps).to(dtype=tensor.dtype)
        return n
