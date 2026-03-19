"""Unit tests for gaussian timestep weighting/sampling."""

import os
import sys

import torch

# Add project root to sys.path for `import toolkit...` when running tests from repo root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.timestep_sampler import TimestepSampler
from extensions_built_in.sd_trainer.gaussian_timestep_weights import evaluate_gaussian_timestep


class _DummyTrainConfig:
    # Ensure we hit the gaussian branch inside `TimestepSampler.sample()`.
    timestep_type = "shift"

    num_train_timesteps = 1000
    gaussian_mean = 450.0
    gaussian_std = 0.45
    gaussian_std_target = None

    # Used only when gaussian_std_target is not None.
    steps = 1000


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

