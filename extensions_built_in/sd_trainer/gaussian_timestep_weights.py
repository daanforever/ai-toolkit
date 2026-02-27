"""
Gaussian timestep weights for loss scaling and timestep sampling.
Weights lie in [0, 1], normalized by maximum (peak at t = mu).
Caches precomputed weights per (num_train_timesteps, mu, sigma, first_ts, last_ts).
Uses t from scheduler_timesteps (positions are indices into the schedule).
"""
import torch

# Module-level cache: dict key -> weights tensor (length ntt+1, CPU float32).
# Key: (ntt, mu, sigma, first_ts, last_ts).
_cache = {}


def evaluate_gaussian_timestep(
    timesteps,
    mu,
    sigma,
    device,
    dtype,
    num_train_timesteps,
    scheduler_timesteps: torch.Tensor,
):
    """
    Return Gaussian weights in [0, 1] for the given timesteps.

    Weights are computed as exp(-0.5 * ((t - mu) / sigma)^2), normalized by the maximum.
    t is taken from scheduler_timesteps (timesteps are indices into the schedule),
    so gaussian_mean targets the actual timestep value (e.g. 800), not the index.

    Args:
        timesteps: 1D tensor of indices into scheduler_timesteps.
        mu: Gaussian mean in [0, 999] timestep space (e.g. 800 for high noise focus).
        sigma: Gaussian std in [0, 1] (e.g. 0.2).
        device: Target device for the returned tensor.
        dtype: Target dtype for the returned tensor.
        num_train_timesteps: Number of diffusion timesteps (e.g. 1000).
        scheduler_timesteps: 1D tensor of schedule values (e.g. scheduler.timesteps).

    Returns:
        1D tensor of weights, same shape as timesteps, on the given device and dtype.
    """
    global _cache
    ntt = int(num_train_timesteps)
    mu_normalized = float(mu) / 999.0
    sigma = float(sigma)

    st = scheduler_timesteps
    first_last = (st[0].cpu().item(), st[-1].cpu().item()) if st.numel() else (0, 0)
    cache_key = (ntt, mu_normalized, sigma, first_last[0], first_last[1])

    if cache_key not in _cache:
        st_f = scheduler_timesteps.float().cpu()
        t = st_f[: ntt + 1].clone()
        t = t / float(ntt)
        if t.numel() < ntt + 1:
            t = torch.cat([t, t[-1:].expand(ntt + 1 - t.numel())])
        raw = torch.exp(-0.5 * ((t - mu_normalized) / sigma) ** 2)
        weights = (raw / raw.max().clamp(min=1e-8)).clone()
        _cache[cache_key] = weights

    cached_weights = _cache[cache_key]
    max_idx = cached_weights.shape[0] - 1
    idx = timesteps.long().clamp(0, max_idx).cpu()
    return cached_weights[idx].to(device=device, dtype=dtype)
