"""Approximate PiSSA-style initialization for LoRA ``lora_down`` (Linear only)."""

from __future__ import annotations

from typing import Optional

import torch


_DEFAULT_POWER_ITERS = 4


@torch.no_grad()
def compute_pissa_linear_lora_down(
    weight: torch.Tensor,
    rank: int,
    *,
    niter: int = _DEFAULT_POWER_ITERS,
) -> Optional[torch.Tensor]:
    """
    Build an ``(rank, in_features)`` matrix for ``nn.Linear(in_features, rank).weight`` from
    base layer weight ``(out_features, in_features)`` using subspace iteration + a small
    ``rank × rank`` eigendecomposition.

    Returns ``None`` if initialization cannot be computed (invalid shapes, numerics, etc.).
    """
    try:
        if weight.dim() != 2:
            return None
        out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
        if rank <= 0 or out_f == 0 or in_f == 0:
            return None
        if rank > min(out_f, in_f):
            return None

        device = weight.device
        dtype_compute = torch.float32
        W = weight.to(dtype=dtype_compute, device=device)

        Q = torch.randn(in_f, rank, device=device, dtype=dtype_compute)
        Q, _ = torch.linalg.qr(Q, mode="reduced")

        for _ in range(max(1, int(niter))):
            Y = W @ Q
            Z = W.mT @ Y
            Q, _ = torch.linalg.qr(Z, mode="reduced")

        Y = W @ Q
        M = Y.mT @ Y
        evals, evecs = torch.linalg.eigh(M)
        evals = evals.flip(0)
        evecs = evecs.flip(1)
        lam = torch.clamp(evals, min=0.0)
        s = torch.sqrt(lam)
        QV = Q @ evecs
        lora_down = s.unsqueeze(1) * QV.mT
        if not torch.isfinite(lora_down).all():
            return None
        return lora_down
    except Exception:
        return None
