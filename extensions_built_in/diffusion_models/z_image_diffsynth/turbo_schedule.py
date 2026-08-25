"""Shared Z-Image Turbo timestep grid (sampling + train)."""

import math
from typing import Optional, Tuple

import torch

from toolkit.samplers.custom_flowmatch_sampler import (
    calculate_shift,
    flowmatch_image_seq_len,
)

from .scheduler_config import STATIC_SHIFT, DYNAMIC_SHIFT_DEFAULTS


def get_turbo_sigmas_and_timesteps(
    num_inference_steps: int,
    denoising_strength: float = 1.0,
    use_dynamic_shifting: bool = False,
    latent_h: Optional[int] = None,
    latent_w: Optional[int] = None,
    shift: float = STATIC_SHIFT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Z-Image Turbo sigma/timestep grid (static shift or Flux-style dynamic)."""
    sigma_min = 0.0
    sigma_max = 1.0
    num_train_timesteps = 1000
    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
    if use_dynamic_shifting and latent_h is not None and latent_w is not None:
        image_seq_len = flowmatch_image_seq_len(latent_h, latent_w)
        mu = calculate_shift(
            image_seq_len,
            DYNAMIC_SHIFT_DEFAULTS["base_image_seq_len"],
            DYNAMIC_SHIFT_DEFAULTS["max_image_seq_len"],
            DYNAMIC_SHIFT_DEFAULTS["base_shift"],
            DYNAMIC_SHIFT_DEFAULTS["max_shift"],
        )
        t = sigmas.clamp(min=1e-8)
        exp_mu = math.exp(mu)
        sigmas = exp_mu / (exp_mu + (1 / t - 1) ** 1)
    else:
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * num_train_timesteps
    return sigmas, timesteps


def turbo_slot_dsigma_weights(n: int) -> torch.Tensor:
    """Normalized |Δσ| slot weights from the same Turbo sigmas as train/sample.

    Last Δσ is ``sigmas[-1] - 0`` (heaviest slot, σ≈0.30→0).
    """
    sigmas, _ = get_turbo_sigmas_and_timesteps(
        num_inference_steps=n,
        use_dynamic_shifting=False,
    )
    # Adjacent drops; last step goes to σ=0.
    next_sigma = torch.cat([sigmas[1:], sigmas.new_zeros(1)])
    dsigma = (sigmas - next_sigma).abs()
    return dsigma / dsigma.sum()


def turbo_slot_sampling_weights(n: int, content_or_style: str) -> torch.Tensor:
    """Slot sampling weights from dsigma, optionally reversed for content.

    balanced/style = dsigma (last slot heaviest, ~30% on 8-step Turbo).
    content = reversed dsigma (first slot heaviest).
    """
    dsigma = turbo_slot_dsigma_weights(n)
    if content_or_style == "content":
        return dsigma.flip(0)
    return dsigma
