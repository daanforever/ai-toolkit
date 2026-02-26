"""
Gaussian timestep weights for loss scaling and timestep sampling.
Weights lie in [0, 1], normalized by maximum (peak at t = mu).
Caches precomputed weights per (num_train_timesteps, mu, sigma) or per schedule.
Uses t = timestep / num_train_timesteps (no inversion), or t from scheduler_timesteps when provided.
"""
import torch
from typing import Optional

# Module-level cache: dict key -> weights tensor (length ntt+1, CPU float32).
# Key: (ntt, mu, sigma) when scheduler_timesteps is None; (ntt, mu, sigma, first_ts, last_ts) when provided.
_cache = {}


def evaluate_gaussian_timestep(
    timesteps,
    mu,
    sigma,
    device,
    dtype,
    num_train_timesteps,
    scheduler_timesteps: Optional[torch.Tensor] = None,
):
    """
    Return Gaussian weights in [0, 1] for the given timesteps.

    Weights are computed as exp(-0.5 * ((t - mu) / sigma)^2), normalized by the maximum.
    Two modes:
    - scheduler_timesteps is None: t = timesteps / num_train_timesteps (positions are timestep values).
    - scheduler_timesteps provided: t is taken from scheduler_timesteps (positions are indices into the schedule).
      Use this for sampling so that gaussian_mean targets the actual timestep value (e.g. 800), not the index.

    Args:
        timesteps: 1D tensor of positions (timestep values or indices into scheduler_timesteps).
        mu: Gaussian mean in [0, 1] (e.g. 0.8).
        sigma: Gaussian std (e.g. 0.2).
        device: Target device for the returned tensor.
        dtype: Target dtype for the returned tensor.
        num_train_timesteps: Number of diffusion timesteps (e.g. 1000).
        scheduler_timesteps: Optional 1D tensor of schedule values (e.g. scheduler.timesteps). If set, weights are in timestep-value space.

    Returns:
        1D tensor of weights, same shape as timesteps, on the given device and dtype.
    """
    global _cache
    ntt = int(num_train_timesteps)
    mu = float(mu)
    sigma = float(sigma)

    if scheduler_timesteps is None:
        cache_key = (ntt, mu, sigma)
    else:
        st = scheduler_timesteps
        first_last = (st[0].cpu().item(), st[-1].cpu().item()) if st.numel() else (0, 0)
        cache_key = (ntt, mu, sigma, first_last[0], first_last[1])

    if cache_key not in _cache:
        if scheduler_timesteps is None:
            timestep_vals = torch.arange(ntt + 1, dtype=torch.float32)
            t = timestep_vals / float(ntt)
        else:
            st = scheduler_timesteps.float().cpu()
            t = st[: ntt + 1].clone()
            t = t / float(ntt)
            if t.numel() < ntt + 1:
                t = torch.cat([t, t[-1:].expand(ntt + 1 - t.numel())])
        raw = torch.exp(-0.5 * ((t - mu) / sigma) ** 2)
        weights = (raw / raw.max().clamp(min=1e-8)).clone()
        _cache[cache_key] = weights

    cached_weights = _cache[cache_key]
    max_idx = cached_weights.shape[0] - 1
    idx = timesteps.long().clamp(0, max_idx).cpu()
    return cached_weights[idx].to(device=device, dtype=dtype)
