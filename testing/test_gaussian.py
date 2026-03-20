"""Unit tests for gaussian timestep weighting/sampling."""

import os
import sys

import torch

# Add project root to sys.path for `import toolkit...` when running tests from repo root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.timestep_sampler import TimestepSampler
from extensions_built_in.sd_trainer.gaussian_timestep_weights import (
    evaluate_gaussian_timestep,
    evaluate_gaussian_timestep_bimodal,
)


class _DummyTrainConfig:
    # Ensure we hit the gaussian branch inside `TimestepSampler.sample()`.
    timestep_type = "shift"

    num_train_timesteps = 1000
    gaussian_mean = 450.0
    gaussian_std = 0.45
    gaussian_mean_2 = 750.0
    gaussian_std_2 = 0.35


class _DummyNoiseScheduler:
    def __init__(self, timesteps: torch.Tensor):
        self.timesteps = timesteps


def test_evaluate_gaussian_timestep_peak_near_mean():
    ntt = 1000
    mu = 450.0
    sigma = 0.45

    timesteps = torch.arange(ntt, dtype=torch.float32)
    weights = evaluate_gaussian_timestep(
        timesteps=timesteps,
        mu=mu,
        sigma=sigma,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_train_timesteps=ntt,
    )
    probs = weights / weights.sum().clamp(min=1e-8)

    # Peak should be very close to `mu` (discrete grid).
    argmax_t = int(torch.argmax(probs).item())
    assert abs(argmax_t - int(mu)) <= 1

    # Probabilities should be normalized.
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)


def test_timestep_sampler_gaussian_matches_expected_mean():
    torch.manual_seed(0)

    ntt = 1000
    mu = 450.0
    sigma = 0.45

    # Choose a realistic denoising range so indices stay within scheduler bounds.
    min_noise_steps = 1
    max_noise_steps = 999

    # Fake scheduler where timestep values equal their indices: timesteps[i] = i.
    noise_scheduler = _DummyNoiseScheduler(torch.arange(ntt, dtype=torch.float32))

    train_config = _DummyTrainConfig()
    train_config.gaussian_mean = mu
    train_config.gaussian_std = sigma
    train_config.num_train_timesteps = ntt

    sampler = TimestepSampler(train_config, noise_scheduler)

    batch_size = 30000
    latents = torch.empty((batch_size, 1), device=torch.device("cpu"))

    # `timesteps` is sampled with replacement; mean should be close to the theoretical mean.
    result = sampler.sample(
        batch_size=batch_size,
        latents=latents,
        content_or_style="gaussian",
        min_noise_steps=min_noise_steps,
        max_noise_steps=max_noise_steps,
        num_train_timesteps=ntt,
        device=torch.device("cpu"),
        step_num=0,
    )
    sampled_timesteps = result.timesteps
    sampled_mean = sampled_timesteps.mean().item()

    allowed_start = ntt - max_noise_steps
    allowed_end = ntt - min_noise_steps
    allowed_indices = torch.arange(allowed_start, allowed_end + 1, dtype=torch.long)
    allowed_timestep_values = noise_scheduler.timesteps[allowed_indices]

    weights = evaluate_gaussian_timestep(
        timesteps=allowed_timestep_values,
        mu=mu,
        sigma=sigma,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_train_timesteps=ntt,
    )
    probs = weights / weights.sum().clamp(min=1e-8)
    expected_mean = (allowed_timestep_values * probs).sum().item()

    # Monte-Carlo tolerance: should be tight enough to catch regressions, loose enough to be stable.
    assert abs(sampled_mean - expected_mean) < 2.0


def test_evaluate_gaussian_timestep_bimodal_two_peaks():
    ntt = 1000
    mu1, mu2 = 200.0, 800.0
    s1, s2 = 0.12, 0.15
    timesteps = torch.arange(ntt, dtype=torch.float32)
    weights = evaluate_gaussian_timestep_bimodal(
        timesteps=timesteps,
        mu1=mu1,
        sigma1=s1,
        mu2=mu2,
        sigma2=s2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_train_timesteps=ntt,
    )
    # Global max-normalization can make one shoulder outrank the second peak in raw top-k;
    # check argmax in windows around each configured mean.
    lo1, hi1 = max(0, int(mu1) - 80), min(ntt, int(mu1) + 80)
    lo2, hi2 = max(0, int(mu2) - 80), min(ntt, int(mu2) + 80)
    peak1_idx = lo1 + int(torch.argmax(weights[lo1:hi1]).item())
    peak2_idx = lo2 + int(torch.argmax(weights[lo2:hi2]).item())
    assert abs(peak1_idx - mu1) <= 25
    assert abs(peak2_idx - mu2) <= 25


def test_timestep_sampler_gaussian_bimodal_expected_mean():
    torch.manual_seed(1)
    ntt = 1000
    mu1, mu2 = 250.0, 750.0
    s1, s2 = 0.15, 0.18
    min_noise_steps = 1
    max_noise_steps = 999
    noise_scheduler = _DummyNoiseScheduler(torch.arange(ntt, dtype=torch.float32))
    train_config = _DummyTrainConfig()
    train_config.gaussian_mean = mu1
    train_config.gaussian_std = s1
    train_config.gaussian_mean_2 = mu2
    train_config.gaussian_std_2 = s2
    train_config.num_train_timesteps = ntt
    sampler = TimestepSampler(train_config, noise_scheduler)
    batch_size = 40000
    latents = torch.empty((batch_size, 1), device=torch.device("cpu"))
    result = sampler.sample(
        batch_size=batch_size,
        latents=latents,
        content_or_style="gaussian_bimodal",
        min_noise_steps=min_noise_steps,
        max_noise_steps=max_noise_steps,
        num_train_timesteps=ntt,
        device=torch.device("cpu"),
        step_num=0,
    )
    allowed_start = ntt - max_noise_steps
    allowed_end = ntt - min_noise_steps
    allowed_indices = torch.arange(allowed_start, allowed_end + 1, dtype=torch.long)
    allowed_timestep_values = noise_scheduler.timesteps[allowed_indices]
    w = evaluate_gaussian_timestep_bimodal(
        allowed_timestep_values,
        mu1,
        s1,
        mu2,
        s2,
        torch.device("cpu"),
        torch.float32,
        ntt,
    )
    probs = w / w.sum().clamp(min=1e-8)
    expected_mean = (allowed_timestep_values * probs).sum().item()
    assert abs(result.timesteps.mean().item() - expected_mean) < 3.0


def test_bimodal_identical_peaks_matches_unimodal_weights():
    """50/50 mixture of the same component equals that component (weights on grid)."""
    ntt = 1000
    mu, sigma = 412.0, 0.33
    t = torch.arange(ntt, dtype=torch.float32)
    w_uni = evaluate_gaussian_timestep(
        t, mu, sigma, torch.device("cpu"), torch.float32, ntt
    )
    w_bi = evaluate_gaussian_timestep_bimodal(
        t, mu, sigma, mu, sigma, torch.device("cpu"), torch.float32, ntt
    )
    assert torch.allclose(w_uni, w_bi, rtol=0, atol=1e-6)


def test_evaluate_gaussian_clamps_timestep_values_to_grid():
    """Scheduler values outside 0..ntt-1 must not crash; lookup clamps to grid ends."""
    ntt = 500
    mu, sigma = 200.0, 0.2
    weird = torch.tensor([-50.0, 0.0, 250.0, 999.0, 5000.0], dtype=torch.float32)
    w = evaluate_gaussian_timestep(
        weird, mu, sigma, torch.device("cpu"), torch.float32, ntt
    )
    t_grid = torch.arange(ntt, dtype=torch.float32)
    w_full = evaluate_gaussian_timestep(
        t_grid, mu, sigma, torch.device("cpu"), torch.float32, ntt
    )
    assert torch.isclose(w[0], w_full[0])
    assert torch.isclose(w[1], w_full[0])
    assert torch.isclose(w[2], w_full[250])
    assert torch.isclose(w[3], w_full[ntt - 1])
    assert torch.isclose(w[4], w_full[ntt - 1])


def test_timestep_sampler_gaussian_bimodal_narrow_window_stays_in_bounds():
    """When the allowed index range is tiny, sampling still yields valid indices."""
    torch.manual_seed(2)
    ntt = 1000
    noise_scheduler = _DummyNoiseScheduler(torch.arange(ntt, dtype=torch.float32))
    train_config = _DummyTrainConfig()
    train_config.gaussian_mean = 100.0
    train_config.gaussian_std = 0.1
    train_config.gaussian_mean_2 = 900.0
    train_config.gaussian_std_2 = 0.1
    train_config.num_train_timesteps = ntt
    sampler = TimestepSampler(train_config, noise_scheduler)
    # Only indices 400..600 → both sharp peaks are outside; weights flat-ish but valid.
    result = sampler.sample(
        batch_size=500,
        latents=torch.empty((500, 1), device=torch.device("cpu")),
        content_or_style="gaussian_bimodal",
        min_noise_steps=400,
        max_noise_steps=600,
        num_train_timesteps=ntt,
        device=torch.device("cpu"),
        step_num=0,
    )
    lo, hi = ntt - 600, ntt - 400
    assert result.timesteps.min() >= noise_scheduler.timesteps[lo]
    assert result.timesteps.max() <= noise_scheduler.timesteps[hi]

