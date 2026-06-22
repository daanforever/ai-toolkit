"""Adafactor step decomposition metrics: effective_lr, precond_gain, momentum_gain."""
import math

import torch
import torch.nn as nn

from toolkit.optimizers.adafactor import Adafactor


def _make_opt(beta1=0.9, **kwargs):
    p = nn.Parameter(torch.full((8, 16), 0.01))
    defaults = dict(
        lr=1e-4,
        beta1=beta1,
        beta2=0.99,
        scale_parameter=True,
        relative_step=True,
        weight_decay=0.0,
    )
    defaults.update(kwargs)
    return Adafactor([p], **defaults), p


def _step_with_grad(opt, p, scale):
    d = torch.randn_like(p)
    d = d / d.norm()
    p.grad = (d * scale).clone()
    opt.step()


def test_effective_lr_positive_after_warmup():
    opt, p = _make_opt()
    for _ in range(20):
        _step_with_grad(opt, p, 1e-3)
    assert opt.get_avg_effective_lr() > 0.0
    assert opt.get_avg_precond_gain() > 0.0


def test_momentum_gain_one_when_beta1_none():
    opt, p = _make_opt(beta1=None)
    for _ in range(10):
        _step_with_grad(opt, p, 1e-3)
    assert math.isclose(opt.get_avg_momentum_gain(), 1.0, rel_tol=1e-5)


def test_decomposition_approx_one_step():
    opt, p = _make_opt()
    for _ in range(30):
        _step_with_grad(opt, p, 1e-3)
    _step_with_grad(opt, p, 1e-5)

    eff = opt.get_avg_effective_lr()
    pre = opt.get_avg_precond_gain()
    mom = opt.get_avg_momentum_gain()
    lr = opt.get_avg_learning_rate()

    product = pre * lr * mom
    assert eff > 0
    assert math.isclose(eff, product, rel_tol=0.15)


def test_step_metrics_stored_as_float_not_tensor():
    opt, p = _make_opt()
    _step_with_grad(opt, p, 1e-4)
    st = opt.state[p]
    for key in ("effective_lr", "precond_gain", "momentum_gain"):
        assert key in st
        assert isinstance(st[key], float)
    assert "update_hat" not in st
    assert "scaled_update" not in st
