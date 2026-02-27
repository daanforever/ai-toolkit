"""
Gaussian timestep weights for loss scaling and timestep sampling.
Weights lie in [0, 1], normalized by maximum (peak at t = mu).
Caches precomputed weights per (num_train_timesteps, mu, sigma).
Works with timestep values directly (0-999), not indices.
"""
import torch

# Module-level cache: dict key -> weights tensor (length ntt+1, CPU float32).
# Key: (ntt, mu_normalized, sigma).
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
    Return Gaussian weights in [0, 1] for the given timestep values.

    Weights are computed as exp(-0.5 * ((t - mu) / sigma)^2), normalized by the maximum.
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
    
    cache_key = (ntt, mu_normalized, sigma)
    
    if cache_key not in _cache:
        t = torch.arange(ntt, dtype=torch.float32) / float(ntt)
        raw = torch.exp(-0.5 * ((t - mu_normalized) / sigma) ** 2)
        weights = (raw / raw.max().clamp(min=1e-8)).clone()
        _cache[cache_key] = weights
    
    cached_weights = _cache[cache_key]
    max_idx = cached_weights.shape[0]
    idx = timesteps.long().clamp(0, max_idx).cpu()
    return cached_weights[idx].to(device=device, dtype=dtype)
