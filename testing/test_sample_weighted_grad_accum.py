"""Unit tests for sample-weighted gradient_accumulation microbatch scales.

Run (repo root, venv):
  venv\\Scripts\\python.exe -m pytest testing/test_sample_weighted_grad_accum.py -q
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.sd_trainer.SDTrainer import (  # noqa: E402
    SDTrainer,
    batch_num_samples,
    sample_weighted_microbatch_scales,
)


def _fake_batch(n: int):
    return SimpleNamespace(
        file_items=[SimpleNamespace() for _ in range(n)],
        latents=None,
        tensor=None,
    )


def test_batch_num_samples_file_items():
    assert batch_num_samples(_fake_batch(3)) == 3


def test_batch_num_samples_latents_fallback():
    batch = SimpleNamespace(
        file_items=None,
        latents=torch.zeros(4, 1, 2, 2),
        tensor=None,
    )
    assert batch_num_samples(batch) == 4


def test_batch_num_samples_tensor_fallback():
    batch = SimpleNamespace(
        file_items=[],
        latents=None,
        tensor=torch.zeros(2, 3, 8, 8),
    )
    assert batch_num_samples(batch) == 2


def test_batch_num_samples_default_one():
    batch = SimpleNamespace(file_items=None, latents=None, tensor=None)
    assert batch_num_samples(batch) == 1


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("gradient_accumulation", [1, 2, 3])
def test_equal_microbatch_scales_match_one_over_n(batch_size, gradient_accumulation):
    batch_sizes = [batch_size] * gradient_accumulation
    scales = sample_weighted_microbatch_scales(batch_sizes)
    n = gradient_accumulation
    expected = [1.0 / n] * n
    assert len(scales) == n
    assert pytest.approx(sum(scales), abs=1e-9) == 1.0
    for s, e in zip(scales, expected):
        assert s == pytest.approx(e, abs=1e-9)


def test_uneven_singleton_scales():
    scales = sample_weighted_microbatch_scales([2, 2, 1])
    assert scales == pytest.approx([2 / 5, 2 / 5, 1 / 5], abs=1e-9)


def test_single_batch_scale_one():
    assert sample_weighted_microbatch_scales([1]) == pytest.approx([1.0], abs=1e-9)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("gradient_accumulation", [1, 2, 3])
def test_hook_train_loop_equal_sizes_passes_scales_and_logs_mean(
    batch_size, gradient_accumulation
):
    trainer = object.__new__(SDTrainer)
    trainer.optimizer = MagicMock()
    trainer.sd = SimpleNamespace(
        is_multistage=False,
        trainable_multistage_boundaries=[],
        multistage_boundaries=[],
    )
    trainer.steps_this_boundary = 0
    trainer.current_boundary_index = 0
    trainer.train_config = SimpleNamespace(
        switch_boundary_every=1,
        optimizer="adamw",
        max_grad_norm=1.0,
    )
    trainer.model_config = SimpleNamespace(low_vram=False)
    trainer.is_grad_accumulation_step = False
    trainer.params = [torch.nn.Parameter(torch.zeros(1))]
    trainer.accelerator = MagicMock()
    trainer.ema = None
    trainer.adapter = None
    trainer.embedding = None
    trainer.timer = lambda _name: nullcontext()
    trainer.lr_scheduler = MagicMock()
    trainer.end_of_training_loop = MagicMock()

    n = gradient_accumulation
    losses = [float(i + 1) for i in range(n)]
    seen_scales: list[float] = []

    def _fake_train_single(batch, microbatch_scale: float = 1.0):
        seen_scales.append(float(microbatch_scale))
        idx = len(seen_scales) - 1
        return torch.tensor(losses[idx], dtype=torch.float32)

    trainer.train_single_accumulation = _fake_train_single

    batch_list = [_fake_batch(batch_size) for _ in range(n)]
    out = SDTrainer.hook_train_loop(trainer, batch_list)

    expected_scales = [1.0 / n] * n
    assert seen_scales == pytest.approx(expected_scales, abs=1e-9)
    assert out["loss"] == pytest.approx(sum(losses) / n, abs=1e-6)
    assert isinstance(out, OrderedDict)


def test_hook_train_loop_uneven_sizes_sample_weighted():
    trainer = object.__new__(SDTrainer)
    trainer.optimizer = MagicMock()
    trainer.sd = SimpleNamespace(
        is_multistage=False,
        trainable_multistage_boundaries=[],
        multistage_boundaries=[],
    )
    trainer.steps_this_boundary = 0
    trainer.current_boundary_index = 0
    trainer.train_config = SimpleNamespace(
        switch_boundary_every=1,
        optimizer="adamw",
        max_grad_norm=1.0,
    )
    trainer.model_config = SimpleNamespace(low_vram=False)
    trainer.is_grad_accumulation_step = False
    trainer.params = [torch.nn.Parameter(torch.zeros(1))]
    trainer.accelerator = MagicMock()
    trainer.ema = None
    trainer.adapter = None
    trainer.embedding = None
    trainer.timer = lambda _name: nullcontext()
    trainer.lr_scheduler = MagicMock()
    trainer.end_of_training_loop = MagicMock()

    batch_sizes = [2, 2, 1]
    losses = [1.0, 3.0, 5.0]
    seen_scales: list[float] = []

    def _fake_train_single(batch, microbatch_scale: float = 1.0):
        seen_scales.append(float(microbatch_scale))
        idx = len(seen_scales) - 1
        return torch.tensor(losses[idx], dtype=torch.float32)

    trainer.train_single_accumulation = _fake_train_single

    batch_list = [_fake_batch(n) for n in batch_sizes]
    out = SDTrainer.hook_train_loop(trainer, batch_list)

    assert seen_scales == pytest.approx([2 / 5, 2 / 5, 1 / 5], abs=1e-9)
    expected_log = (2 * 1.0 + 2 * 3.0 + 1 * 5.0) / 5.0
    assert out["loss"] == pytest.approx(expected_log, abs=1e-6)
    # Must differ from equal microbatch mean (would overweight singleton)
    equal_mean = sum(losses) / 3.0
    assert abs(out["loss"] - equal_mean) > 1e-6
