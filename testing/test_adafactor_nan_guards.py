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


def _same_with_nonfinite(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    nonfinite = ~torch.isfinite(a)
    if not torch.equal(nonfinite, ~torch.isfinite(b)):
        return False
    if nonfinite.any() and not torch.equal(torch.isnan(a), torch.isnan(b)):
        return False
    if nonfinite.any() and not torch.equal(torch.isinf(a), torch.isinf(b)):
        return False
    finite = ~nonfinite
    return torch.equal(a[finite], b[finite])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("unsigned", [False, True])
def test_finite_or_eps_inplace_matches_out_of_place(dtype, unsigned):
    """In-place and out-of-place guards agree; finite values stay bit-identical."""
    eps = 1e-3
    src = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 2.0, -4.0, 0.0, 100.0],
        dtype=dtype,
    )
    snapshot = src.clone()
    out = Adafactor._finite_or_eps(src, eps, unsigned=unsigned)
    assert _same_with_nonfinite(src, snapshot)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()

    inplace = src.clone()
    Adafactor._maybe_finite_or_eps_inplace(inplace, eps, unsigned=unsigned)
    assert inplace.dtype == dtype
    assert torch.equal(out, inplace)

    finite = torch.isfinite(snapshot)
    assert torch.equal(out[finite], snapshot[finite])
    assert out[0] > 0
    assert out[1] > 0
    if unsigned:
        assert out[2] > 0
    else:
        assert out[2] < 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_finite_or_eps_leaves_finite_values_bit_identical(dtype):
    src = torch.tensor([0.0, -1.5, 2.0, 100.0], dtype=dtype)
    out = Adafactor._finite_or_eps(src, 1e-3)
    assert out.dtype == dtype
    assert torch.equal(out, src)
    inplace = src.clone()
    Adafactor._maybe_finite_or_eps_inplace(inplace, 1e-3)
    assert torch.equal(inplace, src)


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
    param.grad = torch.zeros_like(param)
    opt.step()
    assert math.isfinite(group["rms_max"].item())
    assert group["rms_max"].item() == pytest.approx(2.0 * decay_rate)


def test_clamp_effective_wd_tensor_and_scalar():
    t = torch.tensor(float("nan"))
    assert Adafactor._clamp_effective_wd(t) == 0.0
    assert Adafactor._clamp_effective_wd(2.0) == pytest.approx(1.0 - 1e-6)
    clamped = Adafactor._clamp_effective_wd(torch.tensor(0.5))
    assert clamped.item() == pytest.approx(0.5)


def test_all_nan_bf16_grad_keeps_param_finite_and_does_not_mutate_grad():
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = Adafactor(
        [param],
        lr=1e-2,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
        stochastic_accumulation=False,
        stochastic_rounding=False,
    )
    param.grad = torch.full_like(param, float("nan"))
    grad_ptr = param.grad.data_ptr()
    opt.step()
    assert param.dtype == torch.bfloat16
    assert torch.isfinite(param.float()).all()
    assert param.grad.data_ptr() == grad_ptr
    assert torch.isnan(param.grad).all()


@pytest.mark.parametrize(
    "weight_decay_mode",
    ["absolute", "constant", "param_rms", "update_rms"],
)
def test_nan_weights_stay_finite_for_each_weight_decay_mode(weight_decay_mode):
    opt, param = _make_opt(weight_decay=0.1, weight_decay_mode=weight_decay_mode, lr=0.1)
    with torch.no_grad():
        param.copy_(torch.tensor([float("nan"), 1.0, 2.0, 3.0]))
    param.grad = torch.ones_like(param)
    opt.step()
    assert torch.isfinite(param).all()
