"""
Gaussian timestep weights for loss scaling and timestep sampling.
Weights lie in [0, 1], normalized by maximum (peak at t = mu).
Caches precomputed weights per (num_train_timesteps, mu, sigma); cache is invalidated when any of these change.
Uses t = timestep / num_train_timesteps (no inversion).
"""
import torch

# Module-level cache: (num_train_timesteps, mu, sigma, weights_tensor) or None. Weights on CPU, float32.
_cache = None


def get_gaussian_timestep_weights(timesteps, mu, sigma, device, dtype, num_train_timesteps):
    """
    Return Gaussian weights in [0, 1] for the given timesteps.

    Weights are computed as exp(-0.5 * ((t - mu) / sigma)^2) with t = timestep / num_train_timesteps,
    normalized by the maximum (so maximum weight is 1 at t = mu). Results are cached per
    (num_train_timesteps, mu, sigma); cache is invalidated when any of these change.

    Args:
        timesteps: 1D tensor of timestep values in [0, num_train_timesteps] (batch dimension).
        mu: Gaussian mean in [0, 1] (e.g. 0.5).
        sigma: Gaussian std (e.g. 0.2).
        device: Target device for the returned tensor.
        dtype: Target dtype for the returned tensor.
        num_train_timesteps: Number of diffusion timesteps (e.g. 1000).

    Returns:
        1D tensor of weights, same shape as timesteps, on the given device and dtype.
    """
    global _cache
    ntt = int(num_train_timesteps)
    mu = float(mu)
    sigma = float(sigma)

    if _cache is None or _cache[0] != ntt or _cache[1] != mu or _cache[2] != sigma:
        # Precompute weights for timesteps 0..ntt on CPU; t = timestep / ntt in [0, 1]
        timestep_vals = torch.arange(ntt + 1, dtype=torch.float32)
        t = timestep_vals / float(ntt)
        raw = torch.exp(-0.5 * ((t - mu) / sigma) ** 2)
        weights = (raw / raw.max().clamp(min=1e-8)).clone()
        _cache = (ntt, mu, sigma, weights)

    _, _, _, cached_weights = _cache
    idx = timesteps.long().clamp(0, ntt).cpu()
    return cached_weights[idx].to(device=device, dtype=dtype)
