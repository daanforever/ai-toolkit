"""Log timestep weights to TensorBoard for diffusion training."""

from __future__ import annotations

from typing import Optional

import torch

_HIST_MAX_BINS = 64
_HEATMAP_T_BINS = 64
_HEATMAP_W_BINS = 64
_RANGE_EPS = 1e-6


def _add_histogram(writer, tag: str, values: torch.Tensor, step_num: int) -> None:
    try:
        writer.add_histogram(tag, values, step_num, max_bins=_HIST_MAX_BINS)
    except TypeError:
        writer.add_histogram(tag, values, step_num)


def _log_batch_stats(writer, prefix: str, x: torch.Tensor, step_num: int) -> None:
    xf = x.float().flatten()
    if xf.numel() == 0:
        return
    writer.add_scalar(f"{prefix}/mean", xf.mean().item(), step_num)
    writer.add_scalar(f"{prefix}/std", xf.std(unbiased=False).item(), step_num)
    writer.add_scalar(f"{prefix}/min", xf.min().item(), step_num)
    writer.add_scalar(f"{prefix}/max", xf.max().item(), step_num)
    qs = torch.quantile(xf, torch.tensor([0.25, 0.5, 0.75], dtype=xf.dtype))
    writer.add_scalar(f"{prefix}/q25", qs[0].item(), step_num)
    writer.add_scalar(f"{prefix}/q50", qs[1].item(), step_num)
    writer.add_scalar(f"{prefix}/q75", qs[2].item(), step_num)


def _log_timestep_weight_heatmap(
    writer,
    t: torch.Tensor,
    w: torch.Tensor,
    step_num: int,
) -> None:
    """
    2D density of (timestep, loss weight) pairs for this batch.

    Image layout: width = binned timestep (low → high), height = binned weight (low → high).
    """
    tf = t.float().flatten()
    wf = w.float().flatten()
    t_min, t_max = tf.min().item(), tf.max().item()
    w_min, w_max = wf.min().item(), wf.max().item()
    if t_max - t_min < _RANGE_EPS:
        t_min -= _RANGE_EPS
        t_max += _RANGE_EPS
    if w_max - w_min < _RANGE_EPS:
        w_min -= _RANGE_EPS
        w_max += _RANGE_EPS

    span_t = t_max - t_min
    span_w = w_max - w_min
    ti = ((tf - t_min) / span_t * (_HEATMAP_T_BINS - 1e-6)).long().clamp(0, _HEATMAP_T_BINS - 1)
    wi = ((wf - w_min) / span_w * (_HEATMAP_W_BINS - 1e-6)).long().clamp(0, _HEATMAP_W_BINS - 1)
    flat = wi * _HEATMAP_T_BINS + ti
    counts = torch.bincount(flat, minlength=_HEATMAP_W_BINS * _HEATMAP_T_BINS).float()
    grid = counts.view(_HEATMAP_W_BINS, _HEATMAP_T_BINS)
    peak = grid.max()
    if peak <= 0:
        norm = grid
    else:
        norm = grid / peak
    chw = norm.unsqueeze(0).expand(3, -1, -1).contiguous()
    writer.add_image("sampled_timestep_to_loss_weight/batch_heatmap", chw, step_num)


def log_timestep_weights(
    writer,
    step_num: int,
    timesteps: torch.Tensor,
    timestep_weights: Optional[torch.Tensor] = None,
    log_every: Optional[int] = None,
    default_weight: float = 1.0,
) -> None:
    """
    Log batch timesteps and per-sample loss weights to TensorBoard.

    writer: SummaryWriter or None (no-op if None).
    step_num: global step for tags.
    timesteps: tensor of timesteps from the loss step (any device/dtype).
    timestep_weights: optional tensor of per-timestep weights. If None, a tensor
        filled with default_weight will be used.
    log_every: if set, log only when step_num % log_every == 0.

    Tags (former ``timestep_weights/*`` names in parentheses):

    - Histogram ``sampled_diffusion_timesteps/batch`` (was ``batch_timesteps``).
    - Histogram ``per_sample_loss_weights/batch`` (was ``batch_weights``).
    - Scalars under ``sampled_diffusion_timesteps/`` and ``per_sample_loss_weights/``:
      mean, std, min, max, q25, q50, q75. Former ``min_timestep`` / ``max_timestep`` /
      ``mean_weight`` correspond to ``sampled_diffusion_timesteps/min|max`` and
      ``per_sample_loss_weights/mean``.
    - Image ``sampled_timestep_to_loss_weight/batch_heatmap``: joint (T, W) density
      for the batch; horizontal axis = timestep bins, vertical = weight bins.
    """
    if writer is None:
        return
    if log_every is not None and step_num % log_every != 0:
        return

    t = timesteps.detach().cpu().flatten()
    if t.numel() == 0:
        return

    if timestep_weights is None:
        w = torch.full_like(t, fill_value=default_weight, dtype=torch.float32)
    else:
        w = timestep_weights.detach().cpu().view(-1)

    _add_histogram(writer, "sampled_diffusion_timesteps/batch", t, step_num)
    _add_histogram(writer, "per_sample_loss_weights/batch", w, step_num)

    _log_batch_stats(writer, "sampled_diffusion_timesteps", t, step_num)
    _log_batch_stats(writer, "per_sample_loss_weights", w, step_num)

    _log_timestep_weight_heatmap(writer, t, w, step_num)
