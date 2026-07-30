"""Smoke tests for HFAdafactor via get_optimizer."""

import pytest
import torch

from toolkit.optimizer import get_optimizer
from toolkit.optimizers.hf_adafactor import HFAdafactor


def _step_once(opt, p):
    p.grad = torch.ones_like(p)
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_hfadafactor_manual_lr():
    p = torch.nn.Parameter(torch.ones(4, 4))
    opt = get_optimizer(
        [p],
        optimizer_type="hfadafactor",
        learning_rate=1e-3,
        optimizer_params={"scale_parameter": False},
    )
    assert isinstance(opt, HFAdafactor)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert opt.param_groups[0]["relative_step"] is False
    _step_once(opt, p)
    assert torch.isfinite(p).all()


def test_hfadafactor_alias_hf_adafactor():
    p = torch.nn.Parameter(torch.ones(2))
    opt = get_optimizer([p], optimizer_type="hf_adafactor", learning_rate=1e-3)
    assert isinstance(opt, HFAdafactor)
    assert opt.param_groups[0]["relative_step"] is False


def test_hfadafactor_lr_zero_relative_schedule():
    p = torch.nn.Parameter(torch.ones(4, 4))
    opt = get_optimizer(
        [p],
        optimizer_type="hfadafactor",
        learning_rate=0,
        optimizer_params={},
    )
    assert isinstance(opt, HFAdafactor)
    assert opt.param_groups[0]["lr"] is None
    assert opt.param_groups[0]["relative_step"] is True
    _step_once(opt, p)
    assert torch.isfinite(p).all()


def test_hfadafactor_lr_none_relative_schedule():
    p = torch.nn.Parameter(torch.ones(4, 4))
    opt = get_optimizer(
        [p],
        optimizer_type="hfadafactor",
        learning_rate=None,
        optimizer_params={},
    )
    assert isinstance(opt, HFAdafactor)
    assert opt.param_groups[0]["lr"] is None
    assert opt.param_groups[0]["relative_step"] is True
    _step_once(opt, p)


def test_hfadafactor_rejects_local_only_beta2():
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(TypeError):
        get_optimizer(
            [p],
            optimizer_type="hfadafactor",
            learning_rate=1e-3,
            optimizer_params={"beta2": 0.99},
        )
