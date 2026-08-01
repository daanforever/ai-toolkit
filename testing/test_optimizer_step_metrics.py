"""OptimizerStepMetrics: generic metrics outside Adafactor / HFAdafactor."""
import pytest
import torch
import torch.nn as nn

from toolkit.optimizers.adafactor import Adafactor
from toolkit.optimizers.hf_adafactor import HFAdafactor
from toolkit.optimizers.optimizer_metrics import OptimizerStepMetrics


def _step_with_metrics(opt, p, metrics, scale=1e-3):
    d = torch.randn_like(p)
    d = d / d.norm()
    p.grad = (d * scale).clone()
    metrics.before_step(opt)
    opt.step()
    metrics.after_step(opt)


def test_optimizer_step_metrics_with_local_adafactor():
    p = nn.Parameter(torch.full((8, 16), 0.01))
    opt = Adafactor(
        [p],
        lr=1e-4,
        beta1=None,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    metrics = OptimizerStepMetrics()
    _step_with_metrics(opt, p, metrics)
    assert metrics.get_mean_grad_rms() > 0.0
    assert metrics.get_mean_update_rms() > 0.0
    assert metrics.get_mean_rms() > 0.0
    assert metrics.get_mean_dynamic_gain() > 0.0
    assert metrics.get_mean_step_efficiency() > 0.0


def test_optimizer_step_metrics_with_hf_adafactor():
    p = nn.Parameter(torch.full((8, 16), 0.01))
    opt = HFAdafactor(
        [p],
        lr=1e-4,
        relative_step=False,
        scale_parameter=False,
    )
    metrics = OptimizerStepMetrics()
    _step_with_metrics(opt, p, metrics)
    assert metrics.get_mean_grad_rms() > 0.0
    assert metrics.get_mean_update_rms() > 0.0
    assert metrics.get_mean_rms() > 0.0
    assert metrics.get_mean_dynamic_gain() > 0.0
    assert metrics.get_mean_step_efficiency() > 0.0


def test_optimizer_step_metrics_named_group_max_rms():
    p = nn.Parameter(torch.full((4, 4), 0.02))
    opt = Adafactor(
        [{"params": [p], "name": "layer0"}],
        lr=1e-4,
        beta1=None,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    metrics = OptimizerStepMetrics()
    _step_with_metrics(opt, p, metrics)
    max_map = metrics.get_max_rms()
    assert "layer0" in max_map
    assert max_map["layer0"] > 0.0
    assert metrics.get_min_rms() > 0.0


def test_optimizer_step_metrics_running_max_decays():
    p = nn.Parameter(torch.full((4, 4), 1.0))
    opt = torch.optim.SGD([p], lr=0.0)
    metrics = OptimizerStepMetrics(rms_max_decay_rate=0.5)
    p.grad = torch.ones_like(p)
    metrics.before_step(opt)
    opt.step()
    metrics.after_step(opt)
    first_grad_max = metrics.get_mean_grad_rms_max()
    assert first_grad_max > 0.0
    p.grad = torch.ones_like(p) * 0.1
    metrics.before_step(opt)
    opt.step()
    metrics.after_step(opt)
    # decayed previous max should dominate tiny new grad
    assert metrics.get_mean_grad_rms_max() == pytest.approx(first_grad_max * 0.5)
