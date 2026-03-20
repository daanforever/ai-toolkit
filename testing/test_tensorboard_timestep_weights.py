"""Tests for toolkit.util.tensorboard_timestep_weights."""

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from toolkit.util.tensorboard_timestep_weights import log_timestep_weights


class _FakeWriter:
    def __init__(self):
        self.histograms: list[tuple] = []
        self.scalars: list[tuple[str, float, int]] = []
        self.images: list[tuple[str, torch.Tensor, int]] = []

    def add_histogram(self, tag, values, step_num, max_bins=None):
        self.histograms.append((tag, values, step_num, max_bins))

    def add_scalar(self, tag, value, step_num):
        self.scalars.append((tag, float(value), step_num))

    def add_image(self, tag, img_tensor, step_num):
        self.images.append((tag, img_tensor, step_num))


def _scalar_map(writer: _FakeWriter) -> dict[str, float]:
    return {tag: val for tag, val, _ in writer.scalars}


def test_log_timestep_weights_tags_and_values():
    t = torch.tensor([100.0, 200.0, 300.0])
    w = torch.tensor([0.5, 1.0, 1.5])
    writer = _FakeWriter()
    log_timestep_weights(writer, step_num=7, timesteps=t, timestep_weights=w)

    hist_tags = {h[0] for h in writer.histograms}
    assert hist_tags == {
        "sampled_diffusion_timesteps/batch",
        "per_sample_loss_weights/batch",
    }

    sm = _scalar_map(writer)
    assert sm["sampled_diffusion_timesteps/min"] == 100.0
    assert sm["sampled_diffusion_timesteps/max"] == 300.0
    assert sm["sampled_diffusion_timesteps/mean"] == 200.0
    assert sm["per_sample_loss_weights/mean"] == 1.0
    assert sm["per_sample_loss_weights/min"] == 0.5
    assert sm["per_sample_loss_weights/max"] == 1.5

    assert len(writer.images) == 1
    tag, img, step = writer.images[0]
    assert tag == "sampled_timestep_to_loss_weight/batch_heatmap"
    assert step == 7
    assert img.shape == (3, 64, 64)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_log_timestep_weights_default_weight():
    t = torch.tensor([1, 2, 3], dtype=torch.long)
    writer = _FakeWriter()
    log_timestep_weights(writer, 0, timesteps=t, timestep_weights=None, default_weight=2.0)
    sm = _scalar_map(writer)
    assert sm["per_sample_loss_weights/mean"] == 2.0
    assert sm["per_sample_loss_weights/min"] == 2.0
    assert sm["per_sample_loss_weights/max"] == 2.0


def test_log_timestep_weights_no_op():
    t = torch.tensor([1.0])
    log_timestep_weights(None, 0, t, None)
    writer = _FakeWriter()
    log_timestep_weights(writer, 1, t, None, log_every=2)
    assert not writer.histograms and not writer.scalars and not writer.images


def test_log_timestep_weights_empty_batch():
    writer = _FakeWriter()
    log_timestep_weights(writer, 0, torch.tensor([]), None)
    assert not writer.histograms and not writer.scalars and not writer.images
