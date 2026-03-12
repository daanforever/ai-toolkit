"""
Truncated Gaussian timestep weights for loss scaling and timestep sampling.
Weights use a truncated normal distribution on [0, 1], normalized by maximum (peak at t = mu).
Caches precomputed weights per (num_train_timesteps, mu, sigma).
Works with timestep values directly (0-999), not indices.
"""
import math
import torch

# Module-level cache: dict key -> weights tensor (length ntt, float32 on device).
# Key: (ntt, mu_normalized, sigma, device).
_cache = {}


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
    global _cache
    ntt = int(num_train_timesteps)
    mu_normalized = float(mu) / 999.0
    sigma = float(sigma)
    
    cache_key = (ntt, mu_normalized, sigma, device)
    
    if cache_key not in _cache:
        t = torch.arange(ntt, dtype=torch.float32, device=device) / float(ntt)
        # Truncated normal on [0, 1]
        z_lower = (0.0 - mu_normalized) / sigma
        z_upper = (1.0 - mu_normalized) / sigma

        # CDF via math.erf: Φ(x) = 0.5 * (1 + erf(x / √2))
        cdf_upper = 0.5 * (1 + math.erf(z_upper / 2**0.5))
        cdf_lower = 0.5 * (1 + math.erf(z_lower / 2**0.5))
        normalization = cdf_upper - cdf_lower

        # Standard normal PDF: φ(x) = exp(-x²/2) / √(2π)
        z = (t - mu_normalized) / sigma
        phi = torch.exp(-0.5 * z**2) / (2 * 3.141592653589793)**0.5

        # Truncated normal PDF (denom: σ * (Φ(b)-Φ(a)) for unit area)
        raw = phi / (sigma * normalization + 1e-8)

        # Scale to [0, 1] by maximum
        weights = (raw / raw.max().clamp(min=1e-8)).clone()
        _cache[cache_key] = weights
    
    cached_weights = _cache[cache_key]
    max_idx = cached_weights.shape[0]
    idx = timesteps.long().clamp(0, max_idx - 1).to(device=device)
    return cached_weights[idx].to(dtype=dtype)
