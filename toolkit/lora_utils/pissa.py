"""Approximate PiSSA-style initialization for LoRA ``lora_down`` (Linear only)."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch

from toolkit.util.debug import is_debug_enabled


_DEFAULT_POWER_ITERS = 4

_PISSA_LINEAR_MODULE_NAMES = frozenset(
    ("Linear", "LoRACompatibleLinear", "QLinear"),
)


def try_init_linear_lora_down_pissa(
    *,
    init_lora_weights: Optional[str],
    org_module_class_name: str,
    full_rank: bool,
    org_weight: torch.Tensor,
    lora_down: torch.nn.Module,
    lora_dim: int,
    in_dim: int,
    out_dim: int,
    network,
    lora_name: str,
) -> bool:
    """
    If config and module type allow PiSSA for a Linear LoRA branch, fill ``lora_down.weight``.
    Returns True when PiSSA was applied; False → caller should use e.g. kaiming_uniform_.
    """
    if (
        init_lora_weights != "pissa"
        or org_module_class_name not in _PISSA_LINEAR_MODULE_NAMES
        or full_rank
    ):
        return False
    pissa_max_rank = min(out_dim, in_dim)
    if lora_dim > pissa_max_rank:
        emit_pissa_rank_cap_notice_once(network)
        return False
    pissa_down, pissa_fail_reason = compute_pissa_linear_lora_down(
        org_weight.detach(), lora_dim
    )
    if pissa_down is not None:
        with torch.no_grad():
            lora_down.weight.copy_(
                pissa_down.to(
                    device=lora_down.weight.device,
                    dtype=lora_down.weight.dtype,
                )
            )
        if is_debug_enabled():
            print(f"LoRA {lora_name}: init_lora_weights=pissa")
        return True
    if is_debug_enabled():
        r = (pissa_fail_reason or "")[:100]
        print(f"PiSSA fail {lora_name}: {r}")
    return False


def emit_pissa_rank_cap_notice_once(network) -> None:
    """One short UserWarning + optional debug line per LoRA network (avoids log spam)."""
    if network is None:
        return
    if getattr(network, "_pissa_rank_cap_notice_done", False):
        return
    network._pissa_rank_cap_notice_done = True
    warnings.warn(
        "PiSSA: lora_dim > min(out,in) on some Linear layers → kaiming there.",
        UserWarning,
        stacklevel=2,
    )
    if is_debug_enabled():
        print("PiSSA: rank>min(W) → kaiming (per-layer logs off).")


@torch.no_grad()
def compute_pissa_linear_lora_down(
    weight: torch.Tensor,
    rank: int,
    *,
    niter: int = _DEFAULT_POWER_ITERS,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """
    Build an ``(rank, in_features)`` matrix for ``nn.Linear(in_features, rank).weight`` from
    base layer weight ``(out_features, in_features)`` using subspace iteration + a small
    ``rank × rank`` eigendecomposition.

    Returns ``(tensor, None)`` on success. On failure returns ``(None, reason)`` where
    ``reason`` is a short diagnostic string (shape/rank/numerics/exception).
    """
    try:
        if weight.dim() != 2:
            return None, f"weight.dim()!=2 (dim={weight.dim()}, shape={tuple(weight.shape)})"
        out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
        if rank <= 0:
            return None, f"rank<=0 (rank={rank})"
        if out_f == 0 or in_f == 0:
            return None, f"zero extent in weight shape (out_f={out_f}, in_f={in_f})"
        if rank > min(out_f, in_f):
            m = min(out_f, in_f)
            return None, f"rank>min(out_f,in_f): rank={rank}, out_f={out_f}, in_f={in_f}, min={m}"

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
            return None, "non-finite values in lora_down after decomposition"
        return lora_down, None
    except Exception as e:
        return None, f"exception {type(e).__name__}: {e}"
