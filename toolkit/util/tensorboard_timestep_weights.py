"""Log timestep weights to TensorBoard for diffusion training."""

from typing import Optional

import torch


def log_timestep_weights(
    writer,
    step_num: int,
    timesteps: torch.Tensor,
    timestep_weights: Optional[torch.Tensor] = None,
    log_every: Optional[int] = None,
    default_weight: float = 1.0,
) -> None:
    """
    Log batch timesteps and their loss weights to TensorBoard.

    writer: SummaryWriter or None (no-op if None).
    step_num: global step for tags.
    timesteps: tensor of timesteps from the loss step (any device/dtype).
    timestep_weights: optional tensor of per-timestep weights. If None, a tensor
        filled with default_weight will be used.
    log_every: if set, log only when step_num % log_every == 0.
    """
    if writer is None:
        return
    if log_every is not None and step_num % log_every != 0:
        return

    t = timesteps.detach().cpu().flatten()
    if timestep_weights is None:
        w = torch.full_like(t, fill_value=default_weight, dtype=torch.float32)
    else:
        w = timestep_weights.detach().cpu().view(-1)

    writer.add_histogram("timestep_weights/batch_timesteps", t, step_num)
    writer.add_histogram("timestep_weights/batch_weights", w, step_num)

    writer.add_scalar("timestep_weights/mean_weight", w.mean().item(), step_num)
    writer.add_scalar("timestep_weights/min_timestep", t.min().item(), step_num)
    writer.add_scalar("timestep_weights/max_timestep", t.max().item(), step_num)
