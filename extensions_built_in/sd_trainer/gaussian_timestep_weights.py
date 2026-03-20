"""
Truncated Gaussian timestep weights for loss scaling and timestep sampling.
Weights use a truncated normal on [0, 1] (normalized by max); `mu` / `sigma` YAML fields
refer to the discrete training **slot axis** 0 .. num_train_timesteps-1.

`evaluate_gaussian_timestep*` index a precomputed table by slot (via `.long()` on the
`timesteps` argument). When `noise_scheduler.timesteps[i]` is not ~`i` (e.g. flow match),
map batch values with `timestep_values_to_slot_indices` first; use
`scheduler_timesteps_align_with_index_grid` to decide. Cached per (ntt, mu, sigma, ...).
"""
import math
from functools import lru_cache

import torch


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

    # Scale to [0, 1] by maximum
    weights = raw / raw.max().clamp(min=1e-8)
    return weights


def evaluate_gaussian_timestep(
    timesteps,
    mu,
    sigma,
    device,
    dtype,
    num_train_timesteps,
):
    """
    Return truncated normal weights in [0, 1] per batch element.

    Weights are the truncated normal PDF on [0, 1] (CDF-normalized), then scaled by the maximum.
    The `timesteps` tensor selects **rows** of the precomputed length-`ntt` table (slot indices
    0 .. ntt-1 after `.long().clamp`). When scheduler timestep *values* differ from slot indices,
    pass mapped indices (see module docstring).

    Args:
        timesteps: 1D tensor of slot indices (float ok; cast to long inside), shape matches output.
        mu: Gaussian mean on the same discrete slot scale as the table (e.g. 700).
        sigma: Gaussian std in [0, 1] (e.g. 0.2).
        device: Target device for the returned tensor.
        dtype: Target dtype for the returned tensor.
        num_train_timesteps: Number of diffusion timesteps (e.g. 1000).

    Returns:
        1D tensor of weights, same shape as timesteps, on the given device and dtype.
    """
    ntt = int(num_train_timesteps)
    mu_normalized = float(mu) / float(ntt - 1)
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
    return raw / raw.max().clamp(min=1e-8)


def evaluate_gaussian_timestep_bimodal(
    timesteps,
    mu1,
    sigma1,
    mu2,
    sigma2,
    device,
    dtype,
    num_train_timesteps,
):
    """
    Bimodal truncated-normal mixture (50/50), weights in [0, 1] with global max 1.
    Same slot-indexing contract for `timesteps` and same `mu` / `sigma` scale as
    `evaluate_gaussian_timestep`.
    """
    ntt = int(num_train_timesteps)
    mu1n = float(mu1) / float(ntt - 1)
    mu2n = float(mu2) / float(ntt - 1)
    device = torch.device(device)
    device_str = str(device)

    cached_weights = _compute_bimodal_weights(
        ntt, mu1n, float(sigma1), mu2n, float(sigma2), device_str
    )
    max_idx = cached_weights.shape[0]
    idx = timesteps.long().clamp(0, max_idx - 1).to(device=device)
    return cached_weights[idx].to(dtype=dtype)
