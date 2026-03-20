"""
Timestep sampling for diffusion model training.
Extracted from BaseSDTrainProcess for modularity and testability.
"""
import random
from dataclasses import dataclass
from typing import List, Optional, Any

import torch

from toolkit.basic import value_map
from extensions_built_in.sd_trainer.gaussian_timestep_weights import (
    evaluate_gaussian_timestep,
    evaluate_gaussian_timestep_bimodal,
)


def allowed_slot_index_range(
    num_train_timesteps: int,
    min_noise_steps: int,
    max_noise_steps: int,
) -> tuple[int, int]:
    """
    Inclusive scheduler slot bounds [lo, hi] for indices into `noise_scheduler.timesteps`.

    `min_denoising_steps` / `max_denoising_steps` map to slot indices via the same
    formulas as historical code, but `hi` is clamped to `num_train_timesteps - 1` so
    `min_denoising_steps == 0` never produces an out-of-range index (previously
    `ntt - 0 == ntt`, invalid for 0-based timesteps of length `ntt`).
    """
    ntt = int(num_train_timesteps)
    last = ntt - 1
    if last < 0:
        return 0, 0
    lo = ntt - int(max_noise_steps)
    hi = ntt - int(min_noise_steps)
    lo = max(0, min(lo, last))
    hi = max(0, min(hi, last))
    if lo > hi:
        lo = hi
    return lo, hi


@dataclass
class TimestepSamplerResult:
    """Result of timestep sampling: final timesteps and optional indices (None for fixed_cycle)."""
    timesteps: torch.Tensor
    timestep_indices: Optional[torch.Tensor]


class TimestepSampler:
    """
    Samples timesteps (or timestep indices) for diffusion training according to
    content_or_style and timestep_type configuration.
    """

    def __init__(self, train_config: Any, noise_scheduler: Any):
        self.train_config = train_config
        self.noise_scheduler = noise_scheduler
        self._fixed_cycle_resolved_timesteps: Optional[List[float]] = None

    def sample(
        self,
        batch_size: int,
        latents: torch.Tensor,
        content_or_style: str,
        min_noise_steps: int,
        max_noise_steps: int,
        num_train_timesteps: int,
        device: torch.device,
        step_num: int,
    ) -> TimestepSamplerResult:
        """
        Sample timesteps or timestep indices for the current batch.

        Returns TimestepSamplerResult with timesteps always set; timestep_indices
        is None only for fixed_cycle strategy.
        """
        if self.train_config.timestep_type == 'next_sample':
            timestep_indices = self._sample_next_sample(
                batch_size, num_train_timesteps, device
            )
            timesteps = self.noise_scheduler.timesteps[timestep_indices.long()]
            return TimestepSamplerResult(timesteps=timesteps, timestep_indices=timestep_indices)
        elif self.train_config.timestep_type == 'one_step':
            timestep_indices = self._sample_one_step(batch_size, device)
            timesteps = self.noise_scheduler.timesteps[timestep_indices.long()]
            return TimestepSamplerResult(timesteps=timesteps, timestep_indices=timestep_indices)
        elif content_or_style in ['style', 'content']:
            timestep_indices = self._sample_content_style(
                batch_size, latents, content_or_style,
                min_noise_steps, max_noise_steps, num_train_timesteps
            )
            timesteps = self.noise_scheduler.timesteps[timestep_indices.long()]
            return TimestepSamplerResult(timesteps=timesteps, timestep_indices=timestep_indices)
        elif content_or_style == 'gaussian':
            timestep_indices = self._sample_gaussian(
                batch_size, latents, min_noise_steps, max_noise_steps,
                num_train_timesteps
            )
            timesteps = self.noise_scheduler.timesteps[timestep_indices.long()]
            return TimestepSamplerResult(timesteps=timesteps, timestep_indices=timestep_indices)
        elif content_or_style == 'gaussian_bimodal':
            timestep_indices = self._sample_gaussian_bimodal(
                batch_size, latents, min_noise_steps, max_noise_steps,
                num_train_timesteps
            )
            timesteps = self.noise_scheduler.timesteps[timestep_indices.long()]
            return TimestepSamplerResult(timesteps=timesteps, timestep_indices=timestep_indices)
        elif content_or_style == 'fixed_cycle':
            timesteps = self._sample_fixed_cycle(batch_size, latents, step_num)
            return TimestepSamplerResult(timesteps=timesteps, timestep_indices=None)
        elif content_or_style == 'balanced':
            timestep_indices = self._sample_balanced(
                batch_size, min_noise_steps, max_noise_steps, device
            )
            timesteps = self.noise_scheduler.timesteps[timestep_indices.long()]
            return TimestepSamplerResult(timesteps=timesteps, timestep_indices=timestep_indices)
        else:
            raise ValueError(f"Unknown content_or_style {content_or_style}")

    def _sample_next_sample(
        self,
        batch_size: int,
        num_train_timesteps: int,
        device: torch.device,
    ) -> torch.Tensor:
        timestep_indices = torch.randint(
            0,
            num_train_timesteps - 2,
            (batch_size,),
            device=device,
        )
        return timestep_indices.long()

    def _sample_one_step(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size,), device=device, dtype=torch.long)

    def _sample_content_style(
        self,
        batch_size: int,
        latents: torch.Tensor,
        content_or_style: str,
        min_noise_steps: int,
        max_noise_steps: int,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        # FlowMatch (e.g. CustomFlowMatchEulerDiscreteScheduler) uses timesteps that run from high noise to low
        # noise in reverse index order, so smaller indices here can correspond to noisier states compared to DDPM.
        ntt = self.train_config.num_train_timesteps
        orig_timesteps = torch.rand((batch_size,), device=latents.device)

        if content_or_style == 'content':
            timestep_indices = (
                (1 - orig_timesteps) ** self.train_config.timestep_bias_exponent
                * self.train_config.num_train_timesteps
            )
        else:
            timestep_indices = (
                orig_timesteps ** self.train_config.timestep_bias_exponent
                * self.train_config.num_train_timesteps
            )

        lo, hi = allowed_slot_index_range(ntt, min_noise_steps, max_noise_steps)
        timestep_indices = value_map(
            timestep_indices,
            0,
            ntt,
            lo,
            hi,
        )
        timestep_indices = timestep_indices.long().clamp(lo, hi)
        timestep_indices.sort()
        return timestep_indices

    def _sample_gaussian(
        self,
        batch_size: int,
        latents: torch.Tensor,
        min_noise_steps: int,
        max_noise_steps: int,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        ntt = self.train_config.num_train_timesteps
        allowed_start, allowed_end = allowed_slot_index_range(
            ntt, min_noise_steps, max_noise_steps
        )
        all_indices = torch.arange(
            allowed_start, allowed_end + 1, device=latents.device, dtype=torch.long
        )
        weights = evaluate_gaussian_timestep(
            all_indices.to(dtype=torch.float32),
            self.train_config.gaussian_mean,
            self.train_config.gaussian_std,
            latents.device,
            torch.float32,
            ntt,
        )
        probs = weights / weights.sum().clamp(min=1e-8)
        sampled_idx = torch.multinomial(probs, batch_size, replacement=True)
        timestep_indices = all_indices[sampled_idx]
        timestep_indices.sort()
        return timestep_indices

    def _sample_gaussian_bimodal(
        self,
        batch_size: int,
        latents: torch.Tensor,
        min_noise_steps: int,
        max_noise_steps: int,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        ntt = self.train_config.num_train_timesteps
        allowed_start, allowed_end = allowed_slot_index_range(
            ntt, min_noise_steps, max_noise_steps
        )
        all_indices = torch.arange(
            allowed_start, allowed_end + 1, device=latents.device, dtype=torch.long
        )
        weights = evaluate_gaussian_timestep_bimodal(
            all_indices.to(dtype=torch.float32),
            self.train_config.gaussian_mean,
            self.train_config.gaussian_std,
            self.train_config.gaussian_mean_2,
            self.train_config.gaussian_std_2,
            latents.device,
            torch.float32,
            ntt,
        )
        probs = weights / weights.sum().clamp(min=1e-8)
        sampled_idx = torch.multinomial(probs, batch_size, replacement=True)
        timestep_indices = all_indices[sampled_idx]
        timestep_indices.sort()
        return timestep_indices

    def _sample_fixed_cycle(
        self,
        batch_size: int,
        latents: torch.Tensor,
        step_num: int,
    ) -> torch.Tensor:
        timestep_list = self.train_config.fixed_cycle_timesteps
        if not timestep_list:
            raise ValueError(
                "content_or_style is 'fixed_cycle' but fixed_cycle_timesteps is empty"
            )
        if self._fixed_cycle_resolved_timesteps is None:
            list_copy = list(timestep_list)
            if self.train_config.fixed_cycle_seed is not None:
                random.Random(self.train_config.fixed_cycle_seed).shuffle(list_copy)
            st = self.noise_scheduler.timesteps
            resolved = []
            for v in list_copy:
                v_t = torch.tensor(v, device=st.device, dtype=st.dtype)
                idx = (torch.abs(st - v_t)).argmin().item()
                resolved.append(st[idx].item())
            self._fixed_cycle_resolved_timesteps = resolved

        resolved = self._fixed_cycle_resolved_timesteps
        idx_cycle = step_num % len(resolved)
        t_val = resolved[idx_cycle]
        st = self.noise_scheduler.timesteps
        return torch.full(
            (batch_size,), t_val, device=latents.device, dtype=st.dtype
        )

    def _sample_balanced(
        self,
        batch_size: int,
        min_noise_steps: int,
        max_noise_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        if min_noise_steps == max_noise_steps:
            timestep_indices = torch.ones(
                (batch_size,), device=device, dtype=torch.long
            ) * min_noise_steps
        else:
            min_idx = min_noise_steps + 1
            max_idx = max_noise_steps - 1
            if self.train_config.noise_scheduler == 'flowmatch':
                min_idx = min_noise_steps
                max_idx = max_noise_steps
            timestep_indices = torch.randint(
                min_idx,
                max_idx,
                (batch_size,),
                device=device,
            )
        return timestep_indices.long()

    def get_fixed_cycle_cache(self) -> Optional[List[float]]:
        """For debug logging when content_or_style is fixed_cycle."""
        return self._fixed_cycle_resolved_timesteps

    def reset_cache(self) -> None:
        """Reset fixed_cycle cache when scheduler changes."""
        self._fixed_cycle_resolved_timesteps = None
