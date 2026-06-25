"""Tests for Adafactor weight_decay_mode behavior."""

import pytest
import torch

from toolkit.optimizers.adafactor import Adafactor


def _make_opt(
    weight_decay_mode: str,
    lr: float = 0.2,
    wd: float = 0.1,
    wd_increment: float = 0.0,
):
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    opt = Adafactor(
        [param],
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=wd,
        weight_decay_increment=wd_increment,
        weight_decay_mode=weight_decay_mode,
    )
    return opt, param


def test_weight_decay_mode_update_rms_uses_update_scale():
    opt, param = _make_opt("update_rms")
    before = param.detach().clone()
    param.grad = torch.zeros_like(param)
    opt.step()
    assert torch.allclose(param, before)


def test_weight_decay_mode_param_rms_uses_parameter_rms():
    opt, param = _make_opt("param_rms", lr=0.2, wd=0.1)
    before = param.detach().clone()
    param.grad = torch.zeros_like(param)
    opt.step()
    expected = before * (1.0 - 0.1 * 1.0)
    assert torch.allclose(param, expected)


def test_weight_decay_mode_absolute_uses_lr():
    opt, param = _make_opt("absolute", lr=0.2, wd=0.1)
    before = param.detach().clone()
    param.grad = torch.zeros_like(param)
    opt.step()
    expected = before * (1.0 - 0.2 * 0.1)
    assert torch.allclose(param, expected)


def test_set_weight_decay_mode_updates_groups():
    opt, _ = _make_opt("absolute")
    opt.set_weight_decay_mode("param_rms")
    assert opt._weight_decay_mode == "param_rms"
    assert all(g["weight_decay_mode"] == "param_rms" for g in opt.param_groups)


def test_weight_decay_increment_applies_after_step():
    opt, param = _make_opt("absolute", lr=0.2, wd=0.1, wd_increment=0.05)
    before_step_1 = param.detach().clone()
    param.grad = torch.zeros_like(param)
    opt.step()

    expected_step_1 = before_step_1 * (1.0 - 0.2 * 0.1)
    assert torch.allclose(param, expected_step_1)
    assert opt.get_weight_decay() == pytest.approx(0.15)

    before_step_2 = param.detach().clone()
    param.grad = torch.zeros_like(param)
    opt.step()

    expected_step_2 = before_step_2 * (1.0 - 0.2 * 0.15)
    assert torch.allclose(param, expected_step_2)
    assert opt.get_weight_decay() == pytest.approx(0.2)


def test_set_weight_decay_increment_updates_groups():
    opt, _ = _make_opt("absolute")
    opt.set_weight_decay_increment(0.01)
    assert opt._weight_decay_increment == 0.01
    assert all(g["weight_decay_increment"] == 0.01 for g in opt.param_groups)


def test_get_weight_decay_returns_current_value():
    opt, _ = _make_opt("absolute", wd=0.123)
    assert opt.get_weight_decay() == pytest.approx(0.123)


def test_invalid_weight_decay_mode_raises():
    with pytest.raises(ValueError):
        _make_opt("unknown_mode")
