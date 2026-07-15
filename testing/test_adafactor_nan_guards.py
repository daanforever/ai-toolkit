"""Tests for Adafactor NaN/Inf guards."""

import math

import pytest
import torch

from toolkit.optimizers.adafactor import Adafactor


def _make_opt(**kwargs):
    defaults = dict(
        lr=0.1,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
    )
    defaults.update(kwargs)
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    opt = Adafactor([param], **defaults)
    return opt, param


def test_finite_or_eps_signed_never_writes_zero():
    eps = 1e-3
    t = torch.tensor([float("nan"), float("inf"), float("-inf"), 2.0])
    out = Adafactor._finite_or_eps(t, eps)
    assert torch.isfinite(out).all()
    assert (out != 0).all()
    assert out[0].item() == pytest.approx(eps)
    assert out[1].item() == pytest.approx(eps)
    assert out[2].item() == pytest.approx(-eps)
    assert out[3].item() == pytest.approx(2.0)


def test_finite_or_eps_unsigned_uses_positive_eps():
    eps = 1e-30
    t = torch.tensor([float("nan"), float("-inf"), 1.0])
    out = Adafactor._finite_or_eps(t, eps, unsigned=True)
    assert torch.isfinite(out).all()
    assert out[0].item() == pytest.approx(eps)
    assert out[1].item() == pytest.approx(eps)


def test_nan_grad_sanitized_params_remain_finite():
    opt, param = _make_opt()
    param.grad = torch.tensor([float("nan"), 1.0, -1.0, 0.5])
    opt.step()
    assert torch.isfinite(param).all()
    assert torch.isfinite(opt.state[param]["exp_avg_sq"]).all()


def test_factored_preconditioner_zero_ema_step_finite():
    param = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float32))
    opt = Adafactor(
        [param],
        lr=0.1,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        factored=True,
    )
    param.grad = torch.zeros(2, 3)
    opt.step()
    assert torch.isfinite(param).all()


def test_nan_weights_healed_with_eps1_not_zero():
    opt, param = _make_opt()
    eps1 = opt.param_groups[0]["eps"][1]
    with torch.no_grad():
        param.copy_(torch.tensor([float("nan"), 1.0, 2.0, 3.0]))
    param.grad = torch.zeros_like(param)
    opt.step()
    assert torch.isfinite(param).all()
    assert param[0].item() == pytest.approx(eps1)
    assert param[0].item() != 0.0


def test_weight_decay_skipped_when_effective_wd_non_finite(monkeypatch):
    opt, param = _make_opt(weight_decay=0.5, weight_decay_mode="absolute", lr=0.2)
    before = param.detach().clone()
    param.grad = torch.zeros_like(param)
    assert Adafactor._clamp_effective_wd(float("nan")) == 0.0
    monkeypatch.setattr(Adafactor, "_clamp_effective_wd", staticmethod(lambda x: 0.0))
    opt.step()
    assert torch.isfinite(param).all()
    assert torch.allclose(param, before)


def test_weight_decay_update_rms_mode_with_nan_update_rms(monkeypatch):
    opt, param = _make_opt(weight_decay=0.5, weight_decay_mode="update_rms", lr=0.2)
    before = param.detach().clone()
    param.grad = torch.zeros_like(param)
    
    call_count = 0
    original_rms = Adafactor._rms
    def mock_rms(tensor):
        nonlocal call_count
        call_count += 1
        if call_count == 4:
            return torch.tensor(float("nan"))
        return original_rms(tensor)
        
    monkeypatch.setattr(Adafactor, "_rms", staticmethod(mock_rms))
    opt.step()
    
    assert call_count >= 4
    assert torch.isfinite(param).all()
    assert torch.allclose(param, before)


def test_group_running_max_not_poisoned_by_non_finite_rms():
    param = torch.nn.Parameter(torch.tensor([float("nan"), 1.0, 1.0, 1.0]))
    opt = Adafactor(
        [param],
        lr=0.1,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
    )
    group = opt.param_groups[0]
    decay_rate = group["rms_max_decay_rate"]
    group["rms_max"] = torch.tensor(2.0)
    group["grad_rms_max"] = torch.tensor(1.5)
    group["update_rms_max"] = torch.tensor(0.5)
    param.grad = torch.zeros_like(param)
    opt.step()
    assert math.isfinite(group["rms_max"].item())
    assert group["rms_max"].item() == pytest.approx(2.0 * decay_rate)
    assert math.isfinite(group["grad_rms_max"].item())
    assert group["grad_rms_max"].item() == pytest.approx(1.5 * decay_rate)
    assert math.isfinite(group["update_rms_max"].item())
    assert group["update_rms_max"].item() == pytest.approx(0.5 * decay_rate)


def test_clamp_effective_wd_tensor_and_scalar():
    t = torch.tensor(float("nan"))
    assert Adafactor._clamp_effective_wd(t) == 0.0
    assert Adafactor._clamp_effective_wd(2.0) == pytest.approx(1.0 - 1e-6)
    clamped = Adafactor._clamp_effective_wd(torch.tensor(0.5))
    assert clamped.item() == pytest.approx(0.5)


def test_effective_beta2_returns_min_on_non_finite_grad_rms():
    group = {"beta2": 0.99, "beta2_adaptive": True, "beta2_min": 0.9, "grad_rms_max": torch.tensor(1.0)}
    beta2 = Adafactor._effective_beta2(group, torch.tensor(float("nan")), eps0=1e-30, step=10)
    assert beta2 == pytest.approx(0.9)
