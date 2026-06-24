"""
Truncated Gaussian timestep weights for loss scaling and timestep sampling.
Weights use a truncated normal on [0, 1] (normalized by max). The lookup table is indexed
by discrete training **slots** 0 .. num_train_timesteps-1.

`evaluate_gaussian_timestep*` take batch `timesteps` as slot indices (after `.long().clamp`).
Optional `noise_scheduler_timesteps`: when set and the grid is *not* aligned with indices
(`schedule[i] ≢ i`, e.g. FlowMatch linspace 1000→1), `mu` / `mu1` / `mu2` are interpreted as
**scheduler values** and mapped to nearest slots; when aligned or when the argument is omitted,
those means are **slot indices** (DDPM-style).

For per-row batch values that are scheduler scalars, map with `timestep_values_to_slot_indices`
before calling `evaluate_*`. Cached per (ntt, mu, sigma, ...).
"""
import math
from functools import lru_cache

import torch


def _normalize_weights_to_unit_interval(raw: torch.Tensor) -> torch.Tensor:
    """
    Normalize arbitrary non-negative weights into [0, 1].

    Guards against NaN/Inf and enforces strict clipping, so downstream code
    never sees values outside the unit interval due to numeric edge cases.
    """
    safe_raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    max_value = safe_raw.max().clamp(min=1e-8)
    return (safe_raw / max_value).clamp_(0.0, 1.0)


def scheduler_timesteps_align_with_index_grid(
    schedule: torch.Tensor,
    ntt: int,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-3,
) -> bool:
    """
    Check whether a scheduler's `timesteps` tensor matches an index grid.

    `evaluate_gaussian_timestep*` build a lookup table along a slot axis 0..ntt-1
    and then do `timesteps.long()` to index that table. This helper verifies
    whether the given scheduler values are numerically equal to their slots:
      schedule[i] ~= i  for all i in [0, ntt-1].

    If this is true, then passing `timesteps` values directly into `evaluate_*`
    is consistent. If it's false (e.g. FlowMatch schedules use values 1000->1),
    callers must map timestep values back to their slot indices.
    """
    schedule = schedule.detach()
    if schedule.numel() != int(ntt):
        return False
    if schedule.numel() == 0:
        return False

    # Compare on CPU to avoid device-specific float quirks.
    schedule_f = schedule.to(device="cpu", dtype=torch.float32)
    expected = torch.arange(int(ntt), device="cpu", dtype=torch.float32)
    return torch.allclose(schedule_f, expected, rtol=rtol, atol=atol)


def timestep_values_to_slot_indices(
    timestep_values: torch.Tensor,
    schedule: torch.Tensor,
    *,
    ntt: int | None = None,
) -> torch.Tensor:
    """
    Map scheduler timestep *values* to slot indices for gaussian lookup.

    For each value `t` in `timestep_values`, returns:
      argmin_j |schedule[j] - t|

    Returned tensor is float (same shape as input) so it can be passed into
    `evaluate_gaussian_timestep*`, which internally does `.long()` indexing.
    """
    if ntt is None:
        ntt = int(schedule.numel())
    else:
        ntt = int(ntt)

    if schedule.numel() != ntt:
        # Still attempt mapping; the argmin will implicitly select among the
        # provided schedule elements. This keeps the function usable even if
        # the caller passes inconsistent `ntt`.
        ntt = int(schedule.numel())

    # Ensure numeric stability and device alignment for the argmin.
    values = timestep_values.to(dtype=torch.float32, device=schedule.device)
    schedule_f = schedule.to(dtype=torch.float32, device=schedule.device)

    # values: [B] or [...], schedule_f: [ntt]
    # diffs: [..., ntt]
    diffs = (values.unsqueeze(-1) - schedule_f.view((1,) * values.dim() + (ntt,))).abs()
    indices = diffs.argmin(dim=-1)
    return indices.to(dtype=torch.float32)


def _resolve_gaussian_mus_to_slots(
    noise_scheduler_timesteps: torch.Tensor | None,
    ntt: int,
    mu1: float,
    mu2: float | None = None,
) -> tuple[float, float | None]:
    """
    If `noise_scheduler_timesteps` is set and misaligned with 0..ntt-1, treat mu1/mu2 as
    values on that schedule and return nearest slot indices; otherwise return mu1/mu2 unchanged.
    """
    if noise_scheduler_timesteps is None:
        return float(mu1), (float(mu2) if mu2 is not None else None)

    sched = noise_scheduler_timesteps
    if scheduler_timesteps_align_with_index_grid(sched, ntt):
        return float(mu1), (float(mu2) if mu2 is not None else None)

    m1 = torch.tensor([float(mu1)], dtype=torch.float32, device=sched.device)
    s1 = float(timestep_values_to_slot_indices(m1, sched, ntt=ntt)[0].item())
    if mu2 is None:
        return s1, None
    m2 = torch.tensor([float(mu2)], dtype=torch.float32, device=sched.device)
    s2 = float(timestep_values_to_slot_indices(m2, sched, ntt=ntt)[0].item())
    return s1, s2


@lru_cache(maxsize=64)
def _compute_weights(ntt, mu_normalized, sigma, device_str):
    """
    Compute truncated normal weights for ntt timesteps. Cached by lru_cache.
    All args must be hashable (int, float, str).
    """
    device = torch.device(device_str)
    t = torch.arange(ntt, dtype=torch.float32, device=device) / float(ntt - 1)
    # Truncated normal on [0, 1]
    z_lower = (0.0 - mu_normalized) / sigma
    z_upper = (1.0 - mu_normalized) / sigma

    # CDF via math.erf: Φ(x) = 0.5 * (1 + erf(x / √2))
    cdf_upper = 0.5 * (1 + math.erf(z_upper / 2**0.5))
    cdf_lower = 0.5 * (1 + math.erf(z_lower / 2**0.5))
    normalization = cdf_upper - cdf_lower

    # Standard normal PDF: φ(x) = exp(-x²/2) / √(2π)
    z = (t - mu_normalized) / sigma
    phi = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)

    # Truncated normal PDF (denom: σ * (Φ(b)-Φ(a)) for unit area)
    raw = phi / (sigma * normalization + 1e-8)

    # Scale to [0, 1] with numeric guards
    weights = _normalize_weights_to_unit_interval(raw)
    return weights


def evaluate_gaussian_timestep(
    timesteps,
    mu,
    sigma,
    device,
    dtype,
    num_train_timesteps,
    *,
    noise_scheduler_timesteps: torch.Tensor | None = None,
):
    """
    Return truncated normal weights in [0, 1] per batch element.

    Weights are the truncated normal PDF on [0, 1] (CDF-normalized), then scaled by the maximum.
    The `timesteps` tensor selects **rows** of the precomputed length-`ntt` table (slot indices
    0 .. ntt-1 after `.long().clamp`). When scheduler timestep *values* differ from slot indices,
    pass mapped indices (see module docstring).

    Args:
        timesteps: 1D tensor of slot indices (float ok; cast to long inside), shape matches output.
        mu: Gaussian mean in slot space, or in scheduler *value* space if
            `noise_scheduler_timesteps` is passed and misaligned (see module docstring).
        sigma: Gaussian std in [0, 1] (e.g. 0.2).
        device: Target device for the returned tensor.
        dtype: Target dtype for the returned tensor.
        num_train_timesteps: Number of diffusion timesteps (e.g. 1000).
        noise_scheduler_timesteps: Optional `noise_scheduler.timesteps` for mean resolution.

    Returns:
        1D tensor of weights, same shape as timesteps, on the given device and dtype.
    """
    ntt = int(num_train_timesteps)
    mu_res, _ = _resolve_gaussian_mus_to_slots(noise_scheduler_timesteps, ntt, mu, None)
    mu_normalized = float(mu_res) / float(ntt - 1)
    sigma = float(sigma)
    device = torch.device(device)
    device_str = str(device)

    cached_weights = _compute_weights(ntt, mu_normalized, sigma, device_str)
    max_idx = cached_weights.shape[0]
    idx = timesteps.long().clamp(0, max_idx - 1).to(device=device)
    return cached_weights[idx].to(dtype=dtype)


@lru_cache(maxsize=64)
def _compute_bimodal_weights(ntt, mu1_normalized, sigma1, mu2_normalized, sigma2, device_str):
    """Mixture of two truncated normals on [0,1], equal weights 0.5/0.5, then scale by global max."""
    device = torch.device(device_str)
    t = torch.arange(ntt, dtype=torch.float32, device=device) / float(ntt - 1)
    s1 = float(sigma1)
    s2 = float(sigma2)
    m1 = float(mu1_normalized)
    m2 = float(mu2_normalized)

    def raw_truncnorm(mu_norm, sigma):
        z_lower = (0.0 - mu_norm) / sigma
        z_upper = (1.0 - mu_norm) / sigma
        cdf_upper = 0.5 * (1 + math.erf(z_upper / 2**0.5))
        cdf_lower = 0.5 * (1 + math.erf(z_lower / 2**0.5))
        normalization = cdf_upper - cdf_lower
        z = (t - mu_norm) / sigma
        phi = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
        return phi / (sigma * normalization + 1e-8)

    raw = 0.5 * raw_truncnorm(m1, s1) + 0.5 * raw_truncnorm(m2, s2)
    return _normalize_weights_to_unit_interval(raw)


def evaluate_gaussian_timestep_bimodal(
    timesteps,
    mu1,
    sigma1,
    mu2,
    sigma2,
    device,
    dtype,
    num_train_timesteps,
    *,
    noise_scheduler_timesteps: torch.Tensor | None = None,
):
    """
    Bimodal truncated-normal mixture (50/50), weights in [0, 1] with global max 1.
    Same slot-indexing contract for `timesteps` and same `mu` / `sigma` resolution rules as
    `evaluate_gaussian_timestep`.
    """
    ntt = int(num_train_timesteps)
    mu1_res, mu2_res = _resolve_gaussian_mus_to_slots(
        noise_scheduler_timesteps, ntt, mu1, mu2
    )
    assert mu2_res is not None
    mu1n = float(mu1_res) / float(ntt - 1)
    mu2n = float(mu2_res) / float(ntt - 1)
    device = torch.device(device)
    device_str = str(device)

    cached_weights = _compute_bimodal_weights(
        ntt, mu1n, float(sigma1), mu2n, float(sigma2), device_str
    )
    max_idx = cached_weights.shape[0]
    idx = timesteps.long().clamp(0, max_idx - 1).to(device=device)
    return cached_weights[idx].to(dtype=dtype)
