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


def test_warmup_final_intermediate_lr_is_applied_before_cleanup():
    """For warmup_steps=2, step2 must still use intermediate warmup_lr, not jump to target."""
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
        factored=False,
    )
    g = opt.param_groups[0]
    eps1 = g["eps"][1]
    expected_step_1_lr = 1.0 * eps1
    expected_step_2_lr = expected_step_1_lr + (1.0 - expected_step_1_lr) / 2.0

    w1.grad = torch.zeros_like(w1)
    opt.step()
    assert g["lr_mean"].item() == pytest.approx(expected_step_1_lr)

    w1.grad = torch.zeros_like(w1)
    opt.step()
    assert g["lr_mean"].item() == pytest.approx(expected_step_2_lr)

    # Cleanup happens on the next step, then base lr is used.
    w1.grad = torch.zeros_like(w1)
    opt.step()
    assert g["lr_mean"].item() == pytest.approx(1.0)


def test_warmup_stops_all_groups_when_any_reaches_target():
    """When one group completes its segment, stop_warmup runs on all other groups."""
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    opt = Adafactor(
        [{"params": [p1]}, {"params": [p2]}],
        lr=1.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=10,
        beta1=None,
        weight_decay=0.0,
    )
    g0, g1 = opt.param_groups
    g0["warmup_steps"] = 2
    g1["warmup_steps"] = 10

    for _ in range(2):
        p1.grad = torch.zeros_like(p1)
        p2.grad = torch.zeros_like(p2)
        opt.step()

    # First group completed its segment; delayed cleanup still pending.
    assert g0.get("warmup_complete_pending_cleanup") is True
    assert "warmup_lr" in g0

    # Second group must be force-stopped immediately (no longer warming).
    assert not g1.get("warmup_active", False)
    assert "warmup_lr" not in g1
    assert "warmup_progress" not in g1
