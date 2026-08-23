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


def schedule_uses_descending_noise_slots(noise_scheduler, train_config=None) -> bool:
    """FlowMatch / shift schedules: slot 0 is max noise, slot ntt-1 is min noise."""
    if train_config is not None and getattr(train_config, "noise_scheduler", None) == "flowmatch":
        return True
    timesteps = getattr(noise_scheduler, "timesteps", None)
    if timesteps is not None and timesteps.numel() >= 2:
        return float(timesteps[0].item()) > float(timesteps[-1].item())
    return False


def allowed_slot_index_range(
    num_train_timesteps: int,
    min_noise_steps: int,
    max_noise_steps: int,
    *,
    descending: bool = False,
) -> tuple[int, int]:
    """
    Inclusive scheduler slot bounds [lo, hi] for indices into `noise_scheduler.timesteps`.

    `min_denoising_steps` / `max_denoising_steps` map to slot indices via the same
    formulas as historical code, but `hi` is clamped to `num_train_timesteps - 1` so
    `min_denoising_steps == 0` never produces an out-of-range index (previously
    `ntt - 0 == ntt`, invalid for 0-based timesteps of length `ntt`).

    When `descending=True` (FlowMatch: slot 0 = noisiest), denoising step 1..N map to
    slots 0..N-1 instead of the DDPM-inverted range.
    """
    ntt = int(num_train_timesteps)
    last = ntt - 1
    if last < 0:
        return 0, 0
    min_ns = int(min_noise_steps)
    max_ns = int(max_noise_steps)
    if descending:
        lo = 0 if min_ns <= 0 else max(0, min_ns - 1)
        hi = last if max_ns <= 0 else min(last, max_ns - 1)
        lo = max(0, min(lo, last))
        hi = max(0, min(hi, last))
        if lo > hi:
            lo = hi
        return lo, hi
    lo = ntt - max_ns
    hi = ntt - min_ns
    lo = max(0, min(lo, last))
    hi = max(0, min(hi, last))
    if lo > hi:
        lo = hi
    return lo, hi


@dataclass
class TimestepSamplerResult:
    """Result of timestep sampling: final timesteps and optional indices (None for fixed_cycle / turbo_prior)."""
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

    def _descending_noise_slots(self) -> bool:
        return schedule_uses_descending_noise_slots(self.noise_scheduler, self.train_config)

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
        is None for fixed_cycle and turbo_prior strategies.
        """
        # turbo_prior must run before content_or_style gaussian branches, which would
        # otherwise steal sampling when both are set in YAML.
        if self.train_config.timestep_type == 'turbo_prior':
            timesteps = self._sample_turbo_prior(batch_size, latents)
            return TimestepSamplerResult(timesteps=timesteps, timestep_indices=None)

        # Gaussian modes are chosen by `content_or_style` and must not be overridden by
        # `timestep_type` in ('next_sample', 'one_step'). Otherwise runtime/UI updates to
        # timestep_type (DiffusionTrainer.apply_runtime_timestep_type) can silently change
        # sampling while loss still uses gaussian_* weighting.
        if content_or_style == 'gaussian':
            timestep_indices = self._sample_gaussian(
                batch_size, latents, min_noise_steps, max_noise_steps,
                num_train_timesteps,
            )
            timesteps = self.noise_scheduler.timesteps[timestep_indices.long()]
            return TimestepSamplerResult(timesteps=timesteps, timestep_indices=timestep_indices)
        if content_or_style == 'gaussian_bimodal':
            timestep_indices = self._sample_gaussian_bimodal(
                batch_size, latents, min_noise_steps, max_noise_steps,
                num_train_timesteps,
            )
            timesteps = self.noise_scheduler.timesteps[timestep_indices.long()]
            return TimestepSamplerResult(timesteps=timesteps, timestep_indices=timestep_indices)

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

    def _sample_turbo_prior(
        self,
        batch_size: int,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Sample float t from the official Turbo NFE grid with Voronoi jitter."""
        from extensions_built_in.diffusion_models.z_image_diffsynth.turbo_schedule import (
            get_turbo_sigmas_and_timesteps,
        )

        n_steps = int(getattr(self.train_config, "turbo_prior_steps", 8) or 8)
        jitter = float(getattr(self.train_config, "turbo_t_jitter", 0.5) or 0.0)
        _, centers = get_turbo_sigmas_and_timesteps(
            num_inference_steps=n_steps,
            use_dynamic_shifting=False,
        )
        centers = centers.to(device=latents.device, dtype=torch.float32)
        n = centers.numel()
        if n == 0:
            raise ValueError("turbo_prior: empty Turbo timestep grid")

        # Voronoi half-widths: first toward next only; last from previous only (not toward 0).
        deltas = torch.empty(n, device=centers.device, dtype=centers.dtype)
        if n == 1:
            deltas[0] = 0.0
        else:
            deltas[0] = (centers[0] - centers[1]) * 0.5
            deltas[-1] = (centers[-2] - centers[-1]) * 0.5
            for i in range(1, n - 1):
                d_prev = (centers[i - 1] - centers[i]) * 0.5
                d_next = (centers[i] - centers[i + 1]) * 0.5
                deltas[i] = torch.minimum(d_prev, d_next)

        slot = torch.randint(0, n, (batch_size,), device=latents.device)
        t_i = centers[slot]
        if jitter == 0.0:
            return t_i
        u = (torch.rand(batch_size, device=latents.device, dtype=centers.dtype) * 2.0 - 1.0) * jitter
        return t_i + u * 2.0 * deltas[slot]

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
        ntt = int(num_train_timesteps)
        orig_timesteps = torch.rand((batch_size,), device=latents.device)

        if content_or_style == 'content':
            timestep_indices = (
                (1 - orig_timesteps) ** self.train_config.timestep_bias_exponent
                * ntt
            )
        else:
            timestep_indices = (
                orig_timesteps ** self.train_config.timestep_bias_exponent
                * ntt
            )

        lo, hi = allowed_slot_index_range(
            ntt, min_noise_steps, max_noise_steps, descending=self._descending_noise_slots()
        )
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
        ntt = int(num_train_timesteps)
        allowed_start, allowed_end = allowed_slot_index_range(
            ntt, min_noise_steps, max_noise_steps, descending=self._descending_noise_slots()
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
            noise_scheduler_timesteps=self.noise_scheduler.timesteps,
            gaussian_shift=getattr(self.train_config, "gaussian_shift", 0.0),
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
        ntt = int(num_train_timesteps)
        allowed_start, allowed_end = allowed_slot_index_range(
            ntt, min_noise_steps, max_noise_steps, descending=self._descending_noise_slots()
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
            noise_scheduler_timesteps=self.noise_scheduler.timesteps,
            gaussian_shift=getattr(self.train_config, "gaussian_shift", 0.0),
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
        if self.train_config.noise_scheduler == 'flowmatch':
            flow_min_idx = 0 if min_noise_steps <= 0 else min_noise_steps - 1
            flow_max_idx = 0 if max_noise_steps <= 0 else max_noise_steps - 1
            if flow_min_idx > flow_max_idx:
                flow_min_idx = flow_max_idx
            if min_noise_steps == max_noise_steps:
                return torch.full(
                    (batch_size,), flow_min_idx, device=device, dtype=torch.long
                )
            timestep_indices = torch.randint(
                flow_min_idx,
                flow_max_idx + 1,
                (batch_size,),
                device=device,
            )
            return timestep_indices.long()

        if min_noise_steps == max_noise_steps:
            timestep_indices = torch.ones(
                (batch_size,), device=device, dtype=torch.long
            ) * min_noise_steps
        else:
            min_idx = min_noise_steps + 1
            max_idx = max_noise_steps - 1
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
