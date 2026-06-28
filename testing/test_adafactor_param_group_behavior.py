"""Behavior-focused tests for Adafactor param-group semantics and safeguards."""

import pytest
import torch

from toolkit.optimizers.adafactor import Adafactor


def _make_two_group_optimizer(**kwargs):
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    opt = Adafactor(
        [
            {"params": [p1], "relative_step": False, "warmup_init": False},
            {"params": [p2], "relative_step": True, "warmup_init": True},
        ],
        lr=1e-3,
        scale_parameter=False,
        warmup_steps=5,
        beta1=None,
        weight_decay=0.0,
        **kwargs,
    )
    return opt, p1, p2


def test_constructor_preserves_per_group_relative_step():
    """Per-group relative_step must survive ctor kwargs (PyTorch-style override)."""
    opt, _, _ = _make_two_group_optimizer(relative_step=False, warmup_init=False)
    assert [g["relative_step"] for g in opt.param_groups] == [False, True]


def test_constructor_preserves_per_group_warmup_init():
    """Per-group warmup_init must survive ctor kwargs (PyTorch-style group override)."""
    opt, _, _ = _make_two_group_optimizer(relative_step=False, warmup_init=False)
    assert [g["warmup_init"] for g in opt.param_groups] == [False, True]


def test_constructor_keeps_global_warmup_init_when_not_overridden():
    """When groups do not specify warmup_init, ctor value applies to all groups."""
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    opt = Adafactor(
        [{"params": [p1]}, {"params": [p2]}],
        lr=1e-3,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=5,
        beta1=None,
        weight_decay=0.0,
    )
    assert [g["warmup_init"] for g in opt.param_groups] == [True, True]


def test_global_lr_applies_warmup_only_to_enabled_groups():
    """PyTorch-style: warmup progresses only for groups with warmup_init=True."""
    opt, p1, p2 = _make_two_group_optimizer(relative_step=False, warmup_init=False)
    # Enable warmup only for second group.
    opt.param_groups[0]["warmup_init"] = False
    opt.param_groups[1]["warmup_init"] = True

    p1.grad = torch.zeros_like(p1)
    p2.grad = torch.zeros_like(p2)
    opt.step()

    assert "warmup_lr" not in opt.param_groups[0]
    assert "warmup_lr" in opt.param_groups[1]


def test_emergency_brake_zero_init_uses_lr_fraction_fallback():
    """With zero-init params, emergency brake must not clamp LR to exact zero."""
    p = torch.nn.Parameter(torch.zeros(8))
    opt = Adafactor(
        [p],
        lr=1e-3,
        relative_step=False,
        scale_parameter=False,
        emergency_brake=0.1,
        beta1=None,
        weight_decay=0.0,
    )
    p.grad = torch.ones_like(p)
    before = p.detach().clone()
    opt.step()

    assert opt.get_mean_learning_rate() == pytest.approx(1e-4, rel=1e-6)
    assert not torch.allclose(p, before)


def test_step_closure_runs_with_grad_enabled():
    """Closure with backward() must work like in standard PyTorch optimizers."""
    p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    opt = Adafactor(
        [p],
        lr=1e-2,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
    )
    grad_enabled_flags = []

    def closure():
        grad_enabled_flags.append(torch.is_grad_enabled())
        opt.zero_grad()
        loss = ((p - 1.0) ** 2).sum()
        loss.backward()
        return loss

    before = p.detach().clone()
    loss = opt.step(closure=closure)

    assert grad_enabled_flags == [True]
    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss.detach(), ((before - 1.0) ** 2).sum())
    assert torch.all(p < before)


def test_load_state_dict_uses_config_priority_except_accumulated_weight_decay():
    """Resume behavior: config wins for static params; checkpoint wins for accumulated weight_decay."""
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    opt_src = Adafactor(
        [
            {"params": [p1], "relative_step": True, "warmup_init": True, "weight_decay": 0.2},
            {"params": [p2], "relative_step": False, "warmup_init": False, "weight_decay": 0.3},
        ],
        lr=1e-3,
        scale_parameter=False,
        beta1=None,
        weight_decay=0.0,  # per-group values above should win
    )
    state = opt_src.state_dict()

    q1 = torch.nn.Parameter(torch.ones(2))
    q2 = torch.nn.Parameter(torch.ones(2))
    opt_dst = Adafactor(
        [
            {"params": [q1], "relative_step": False, "warmup_init": False, "weight_decay": 0.01},
            {"params": [q2], "relative_step": True, "warmup_init": True, "weight_decay": 0.02},
        ],
        lr=1e-3,
        scale_parameter=False,
        beta1=None,
        weight_decay=0.0,  # per-group values above should win
    )
    expected_relative = [g["relative_step"] for g in opt_dst.param_groups]
    expected_warmup = [g["warmup_init"] for g in opt_dst.param_groups]
    expected_wd = [g["weight_decay"] for g in opt_src.param_groups]

    opt_dst.load_state_dict(state)

    assert [g["relative_step"] for g in opt_dst.param_groups] == expected_relative
    assert [g["warmup_init"] for g in opt_dst.param_groups] == expected_warmup
    assert [g["weight_decay"] for g in opt_dst.param_groups] == expected_wd
