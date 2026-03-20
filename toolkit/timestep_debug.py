"""
Timestep distribution debug logging for diffusion training.
Extracted from BaseSDTrainProcess for modularity and testability.
"""
from typing import Any, List, Optional

import torch

from toolkit.print import print_acc
from extensions_built_in.sd_trainer.gaussian_timestep_weights import (
    evaluate_gaussian_timestep,
    evaluate_gaussian_timestep_bimodal,
    scheduler_timesteps_align_with_index_grid,
)


class TimestepDistributionLogger:
    """
    Collects and logs timestep distribution statistics when debug is enabled.
    Used to verify timestep sampling behavior during training.
    """

    def __init__(self, train_config: Any, logging_config: Any, sd: Optional[Any] = None) -> None:
        self.train_config = train_config
        self.logging_config = logging_config
        # Optional Stable Diffusion model reference; when provided, we can
        # introspect the actual noise_scheduler instance for debug printing.
        self.sd = sd
        self._collected_indices: List[Any] = []
        self._collected_timesteps: List[float] = []

    def collect(
        self,
        timestep_indices: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        content_or_style: str,
        step_num: int,
        timestep_sampler: Any,
    ) -> None:
        """Collect indices and timesteps for the current step."""
        if content_or_style == "fixed_cycle":
            cache = timestep_sampler.get_fixed_cycle_cache()
            if cache:
                self._collected_indices.append(step_num % len(cache))
            self._collected_timesteps.extend(timesteps.cpu().tolist())
        else:
            if timestep_indices is not None:
                self._collected_indices.extend(timestep_indices.cpu().tolist())
            self._collected_timesteps.extend(timesteps.cpu().tolist())

    def should_log(self) -> bool:
        """Return True when enough samples have been collected to log."""
        threshold = (self.logging_config.log_every or 0) * 100
        return len(self._collected_indices) >= threshold

    def log_and_reset(
        self,
        step_num: int,
        min_noise_steps: int,
        max_noise_steps: int,
        scheduler_timesteps: torch.Tensor,
    ) -> None:
        """Output collected statistics and clear buffers."""
        threshold = (self.logging_config.log_every or 0) * 100
        num_samples = threshold

        scheduler_timesteps_list = scheduler_timesteps.cpu().tolist()
        content_or_style = self.train_config.content_or_style

        print_acc(f"\n{'='*70}")
        print_acc(f"TIMESTEP DISTRIBUTION DEBUG")
        print_acc(f"{'='*70}")

        print_acc(f"Total scheduler timesteps length: {len(scheduler_timesteps_list)}")

        print_acc(f"\nFirst 10 timestep_indices (generated indices):")
        print_acc(f"{self._collected_indices[:10]}")
        print_acc(f"\nFirst 10 timesteps (actual values after indexing):")
        print_acc(f"{self._collected_timesteps[:10]}")

        weights_list: Optional[List[float]] = None
        if self.train_config.timestep_type == "gaussian":
            ntt = self.train_config.num_train_timesteps
            schedule_aligned = scheduler_timesteps_align_with_index_grid(
                scheduler_timesteps, ntt
            )
            ts_tensor = torch.tensor(
                (
                    self._collected_timesteps[:num_samples]
                    if schedule_aligned
                    else self._collected_indices[:num_samples]
                ),
                device=torch.device("cpu"),
                dtype=torch.long,
            )
            weights_tensor = evaluate_gaussian_timestep(
                ts_tensor,
                self.train_config.gaussian_mean,
                self.train_config.gaussian_std,
                torch.device("cpu"),
                torch.float32,
                ntt,
            )
            weights_list = weights_tensor.tolist()
            pairs_10 = list(zip(self._collected_timesteps[:10], weights_list[:10]))
            print_acc(f"\nFirst 10 (timestep, loss_weight): {pairs_10}")
        elif self.train_config.timestep_type == "gaussian_bimodal":
            ntt = self.train_config.num_train_timesteps
            schedule_aligned = scheduler_timesteps_align_with_index_grid(
                scheduler_timesteps, ntt
            )
            ts_tensor = torch.tensor(
                (
                    self._collected_timesteps[:num_samples]
                    if schedule_aligned
                    else self._collected_indices[:num_samples]
                ),
                device=torch.device("cpu"),
                dtype=torch.long,
            )
            weights_tensor = evaluate_gaussian_timestep_bimodal(
                ts_tensor,
                self.train_config.gaussian_mean,
                self.train_config.gaussian_std,
                self.train_config.gaussian_mean_2,
                self.train_config.gaussian_std_2,
                torch.device("cpu"),
                torch.float32,
                ntt,
            )
            weights_list = weights_tensor.tolist()
            pairs_10 = list(zip(self._collected_timesteps[:10], weights_list[:10]))
            print_acc(f"\nFirst 10 (timestep, loss_weight): {pairs_10}")

        print_acc(f"Config:")
        print_acc(f"  content_or_style: {content_or_style}")
        print_acc(f"  noise_scheduler: {self.train_config.noise_scheduler}")
        # When sd is available, also show the concrete noise scheduler type
        # used by the model (e.g. DiffSynthZImageSchedulerAdapter), since it
        # can differ from the string stored in train_config.noise_scheduler.
        scheduler_type: Optional[str] = None
        sd = getattr(self, "sd", None)
        if sd is not None:
            noise_scheduler = getattr(sd, "noise_scheduler", None)
            if noise_scheduler is not None:
                try:
                    scheduler_type = type(noise_scheduler).__name__
                except Exception:
                    scheduler_type = str(type(noise_scheduler))
        if scheduler_type is not None:
            print_acc(f"  noise_scheduler_obj: {scheduler_type}")
        print_acc(f"  timestep_type: {self.train_config.timestep_type}")
        print_acc(f"  num_train_timesteps: {self.train_config.num_train_timesteps}")
        print_acc(f"  min_denoising_steps: {min_noise_steps}")
        print_acc(f"  max_denoising_steps: {max_noise_steps}")
        print_acc(f"  gaussian_mean: {self.train_config.gaussian_mean}")
        print_acc(f"  gaussian_std: {self.train_config.gaussian_std}")
        print_acc(f"  gaussian_mean_2: {self.train_config.gaussian_mean_2}")
        print_acc(f"  gaussian_std_2: {self.train_config.gaussian_std_2}")

        indices_min = min(self._collected_indices[:num_samples])
        indices_max = max(self._collected_indices[:num_samples])
        indices_mean = sum(self._collected_indices[:num_samples]) / num_samples

        timesteps_min = min(self._collected_timesteps[:num_samples])
        timesteps_max = max(self._collected_timesteps[:num_samples])
        timesteps_mean = sum(self._collected_timesteps[:num_samples]) / num_samples

        print_acc(f"\nStatistics ({num_samples} samples):")
        print_acc(f"  Indices: max={indices_max}, mean={indices_mean:.1f}, min={indices_min}")
        print_acc(
            f"  Timesteps: max={timesteps_max:.1f}, mean={timesteps_mean:.1f}, min={timesteps_min:.1f}"
        )
        if weights_list is not None:
            weights_min = min(weights_list)
            weights_max = max(weights_list)
            weights_mean = sum(weights_list) / num_samples
            print_acc(
                f"  Loss weights: max={weights_max:.3f}, mean={weights_mean:.3f}, min={weights_min:.3f}"
            )
        print_acc(
            f"  Step: {step_num} ({step_num * 100 / self.train_config.steps:.1f}%)"
        )
        print_acc(f"{'='*70}\n")

        self._collected_indices = []
        self._collected_timesteps = []
