"""Unit tests for Adafactor group-level warmup (_global_lr / _warmup_update_group)."""

import pytest
import torch

from toolkit.optimizers.adafactor import Adafactor


def test_warmup_single_increment_per_step_two_params():
    """One optimizer.step must advance group warmup once, not once per parameter."""
    w1 = torch.nn.Parameter(torch.ones(4))
    w2 = torch.nn.Parameter(torch.ones(4))
    opt = Adafactor(
        [w1, w2],
        lr=1.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=10,
        beta1=None,
        weight_decay=0.0,
    )
    w1.grad = torch.zeros_like(w1)
    w2.grad = torch.zeros_like(w2)
    g = opt.param_groups[0]
    eps1 = g["eps"][1]
    opt.step()
    assert g["warmup_progress"] == 1
    assert g["warmup_lr"] == pytest.approx(1.0 * eps1)


def test_warmup_advances_when_no_grad():
    """Warmup ticks once per step() even if no parameter receives an update."""
    w1 = torch.nn.Parameter(torch.ones(2))
    opt = Adafactor(
        [w1],
        lr=0.5,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=5,
        beta1=None,
        weight_decay=0.0,
    )
    g = opt.param_groups[0]
    eps1 = g["eps"][1]
    w1.grad = None
    opt.step()
    assert g["warmup_progress"] == 1
    assert g["warmup_lr"] == pytest.approx(0.5 * eps1)


def test_warmup_no_spurious_segment_after_complete():
    """After warmup finishes, idle steps must not restart a segment (warmup_target vs lr)."""
    w1 = torch.nn.Parameter(torch.ones(1))
    opt = Adafactor(
        [w1],
        lr=1.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=2,
        beta1=None,
        weight_decay=0.0,
    )
    w1.grad = torch.zeros_like(w1)
    g = opt.param_groups[0]
    opt.step()
    opt.step()
    assert not g.get("warmup_active", True)
    assert g["warmup_target"] == 1.0
    opt.step()
    assert not g.get("warmup_active", True)
    assert "warmup_progress" not in g
