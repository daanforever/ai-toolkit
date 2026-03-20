"""
Truncated Gaussian timestep weights for loss scaling and timestep sampling.
Weights use a truncated normal distribution on [0, 1], normalized by maximum (peak at t = mu).
Caches precomputed weights per (num_train_timesteps, mu, sigma).
Works with timestep values directly (0-999), not indices.
"""
import math
from functools import lru_cache

import torch


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
    Return truncated normal weights in [0, 1] for the given timestep values.

    Weights are the truncated normal PDF on [0, 1] (CDF-normalized), then scaled by the maximum.
    Works with actual timestep values (0-999), not scheduler indices.

    Args:
        timesteps: 1D tensor of timestep values in [0, num_train_timesteps].
        mu: Gaussian mean in [0, 999] timestep space (e.g. 700 for high noise focus).
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
    Same timestep / mu / sigma conventions as evaluate_gaussian_timestep.
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
