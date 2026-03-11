"""Log timestep weights to TensorBoard for diffusion training."""

from typing import Optional

import torch


def log_timestep_weights(
    writer,
    step_num: int,
    timesteps: torch.Tensor,
    timestep_weights: torch.Tensor,
    log_every: Optional[int] = None,
) -> None:
    """
    Log batch timesteps and their loss weights to TensorBoard.

    writer: SummaryWriter or None (no-op if None).
    step_num: global step for tags.
    timesteps, timestep_weights: tensors from the loss step (any device/dtype).
    log_every: if set, log only when step_num % log_every == 0.
    """
    if writer is None:
        return
    if log_every is not None and step_num % log_every != 0:
        return

    t = timesteps.detach().cpu().flatten()
    w = timestep_weights.detach().cpu().view(-1)

    writer.add_histogram("timestep_weights/batch_timesteps", t, step_num)
    writer.add_histogram("timestep_weights/batch_weights", w, step_num)

    writer.add_scalar("timestep_weights/mean_weight", w.mean().item(), step_num)
    writer.add_scalar("timestep_weights/min_timestep", t.min().item(), step_num)
    writer.add_scalar("timestep_weights/max_timestep", t.max().item(), step_num)
