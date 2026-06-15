"""Tests for Adafactor weight_decay_mode behavior."""

import pytest
import torch

from toolkit.optimizers.adafactor import Adafactor


def _make_opt(weight_decay_mode: str, lr: float = 0.2, wd: float = 0.1):
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    opt = Adafactor(
        [param],
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=wd,
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


def test_invalid_weight_decay_mode_raises():
    with pytest.raises(ValueError):
        _make_opt("unknown_mode")
