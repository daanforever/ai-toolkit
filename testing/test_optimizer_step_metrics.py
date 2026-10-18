"""OptimizerStepMetrics: generic metrics outside Adafactor / HFAdafactor."""
import math

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


def test_nonfinite_candidate_leaves_finite_running_extrema():
    p = nn.Parameter(torch.ones(4, 4))
    opt = torch.optim.SGD([{"params": [p], "name": "layer0"}], lr=0.1)
    metrics = OptimizerStepMetrics(rms_max_decay_rate=0.5)
    p.grad = torch.ones_like(p)
    metrics.before_step(opt)
    opt.step()
    metrics.after_step(opt)
    finite_grad_max = metrics.get_mean_grad_rms_max()
    finite_min = metrics.get_min_rms()
    finite_named_max = metrics.get_max_rms()["layer0"]
    finite_update_max = metrics.get_mean_update_rms_max()

    with torch.no_grad():
        p.copy_(torch.full_like(p, float("nan")))
    p.grad = torch.full_like(p, float("nan"))
    metrics.before_step(opt)
    opt.step()
    metrics.after_step(opt)

    assert metrics.get_mean_grad_rms_max() == pytest.approx(finite_grad_max * 0.5)
    assert metrics.get_min_rms() == pytest.approx(finite_min / 0.5)
    assert metrics.get_max_rms()["layer0"] == pytest.approx(finite_named_max * 0.5)
    assert metrics.get_mean_update_rms_max() == pytest.approx(finite_update_max * 0.5)


def _backward_twice(param: nn.Parameter) -> None:
    for _ in range(2):
        inputs = torch.randn(4, param.shape[0], dtype=param.dtype, device=param.device)
        (inputs @ param).pow(2).mean().backward()


def _assert_positive_finite(*values: float) -> None:
    for value in values:
        assert math.isfinite(value)
        assert value > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stochastic rounding requires CUDA")
def test_step_metrics_see_stochastically_accumulated_grad():
    param = nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16, device="cuda") * 0.02)
    opt = Adafactor(
        [param],
        lr=1e-3,
        beta1=None,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        stochastic_accumulation=True,
    )
    metrics = OptimizerStepMetrics()
    _backward_twice(param)
    assert param.grad is None
    assert param._accum_grad is not None
    before = param.detach().clone()
    metrics.before_step(opt)
    assert param._accum_grad is not None
    opt.step()
    metrics.after_step(opt)
    assert not hasattr(param, "_accum_grad")
    assert not torch.equal(param.detach(), before)
    _assert_positive_finite(
        metrics.get_mean_grad_rms(),
        metrics.get_mean_grad_rms_max(),
        metrics.get_mean_rms(),
        metrics.get_mean_update_rms(),
        metrics.get_mean_update_rms_max(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stochastic rounding requires CUDA")
def test_step_metrics_with_visible_bf16_grad():
    param = nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16, device="cuda") * 0.02)
    opt = Adafactor(
        [param],
        lr=1e-3,
        beta1=None,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        stochastic_accumulation=False,
    )
    metrics = OptimizerStepMetrics()
    _backward_twice(param)
    assert param.grad is not None
    metrics.before_step(opt)
    opt.step()
    metrics.after_step(opt)
    _assert_positive_finite(
        metrics.get_mean_grad_rms(),
        metrics.get_mean_grad_rms_max(),
        metrics.get_mean_rms(),
        metrics.get_mean_update_rms(),
        metrics.get_mean_update_rms_max(),
    )


def test_before_step_reads_accum_grad_rms():
    param = nn.Parameter(torch.full((4, 4), 0.02))
    accum = torch.full_like(param, 0.1)
    param._accum_grad = accum.clone()
    opt = torch.optim.SGD([param], lr=0.0)
    metrics = OptimizerStepMetrics()
    metrics.before_step(opt)
    expected = OptimizerStepMetrics._rms(accum).item()
    assert metrics.get_mean_grad_rms() == pytest.approx(expected)
    assert metrics.get_mean_rms() > 0.0


def test_first_nonfinite_candidate_reads_as_missing():
    p = nn.Parameter(torch.full((4,), float("nan")))
    opt = torch.optim.SGD([{"params": [p], "name": "bad"}], lr=0.0)
    metrics = OptimizerStepMetrics()
    p.grad = torch.full_like(p, float("inf"))
    metrics.before_step(opt)
    opt.step()
    metrics.after_step(opt)
    assert metrics.get_max_rms() == {}
    assert metrics.get_min_rms() == 0.0
    assert metrics.get_mean_grad_rms_max() == 0.0
    assert metrics.get_mean_update_rms_max() == 0.0
