# Adapter so we use DiffSynth's Z-Image training schedule (timesteps, add_noise, training_weight)
# in our trainer without replacing the whole process.

import torch


def _get_diffsynth_flow_match_scheduler():
    import sys
    import os
    diffynth_root = os.path.join(os.path.dirname(__file__), "DiffSynth-Studio")
    if diffynth_root not in sys.path:
        sys.path.insert(0, diffynth_root)
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    return FlowMatchScheduler(template="Z-Image")


class DiffSynthZImageSchedulerAdapter(torch.nn.Module):
    """
    Wraps DiffSynth's FlowMatchScheduler("Z-Image") so our BatchProcessor and
    base_model.add_noise use the same timesteps/add_noise/training_weight as
    the original Z-Image.sh training loop.
    """

    def __init__(self):
        super().__init__()
        self._scheduler = _get_diffsynth_flow_match_scheduler()
        self._scheduler.set_timesteps(1000, denoising_strength=1.0, training=True)
        self.timesteps = self._scheduler.timesteps
        self.sigmas = self._scheduler.sigmas
        self.linear_timesteps_weights = getattr(
            self._scheduler, "linear_timesteps_weights", torch.ones_like(self._scheduler.timesteps, dtype=torch.float32)
        )
        self.config = type("_Config", (), {"num_train_timesteps": 1000})()

    def set_timesteps(self, num_train_timesteps=1000, device=None, **kwargs):
        self._scheduler.set_timesteps(num_train_timesteps, denoising_strength=1.0, training=True, **kwargs)
        self.timesteps = self._scheduler.timesteps
        self.sigmas = self._scheduler.sigmas
        self.linear_timesteps_weights = getattr(
            self._scheduler, "linear_timesteps_weights", torch.ones_like(self._scheduler.timesteps, dtype=torch.float32)
        )
        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)
            self.linear_timesteps_weights = self.linear_timesteps_weights.to(device)

    def set_train_timesteps(
        self,
        num_timesteps,
        device,
        timestep_type="linear",
        latents=None,
        patch_size=1,
    ):
        self.set_timesteps(num_timesteps, device=device)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self._scheduler.add_noise(original_samples, noise, timesteps)

    def get_timestep_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Return training_weight per batch element (same as DiffSynth FlowMatchSFTLoss)."""
        device = timesteps.device
        dtype = timesteps.dtype
        tw = self.linear_timesteps_weights.to(device=device, dtype=dtype)
        tt = self.timesteps.to(device=device)
        # timesteps can be (B,) or (1,); find index per element
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        indices = torch.argmin(
            (tt.unsqueeze(0) - timesteps.unsqueeze(1)).abs(), dim=1
        )
        return tw[indices]
