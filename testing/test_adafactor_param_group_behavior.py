"""Behavior-focused tests for Adafactor param-group semantics and safeguards."""

import math

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


def test_get_lr_none_uses_base_lr_one():
    """lr=None is treated as base_lr=1.0 in _get_lr (raw-API footgun)."""
    p = torch.nn.Parameter(torch.ones(2, 2))
    opt = Adafactor(
        [p],
        lr=None,
        relative_step=True,
        scale_parameter=False,
        beta1=None,
        weight_decay=0.0,
    )
    opt.param_groups[0]["rms_max"] = torch.tensor(1.0)
    # relative_step does not change LR; parameter RMS does not scale it.
    assert opt._get_lr(opt.param_groups[0]) == pytest.approx(1.0)


def test_get_lr_ignores_parameter_rms_when_scale_parameter():
    """scale_parameter caps the applied step; it does not multiply LR by RMS."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = Adafactor(
        [p],
        lr=1.0,
        relative_step=False,
        scale_parameter=True,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
        stochastic_accumulation=False,
        stochastic_rounding=False,
    )
    assert opt._get_lr(opt.param_groups[0]) == pytest.approx(1.0)
    before = p.detach().clone()
    param_rms = before.pow(2).mean().sqrt().item()
    p.grad = torch.randn_like(p)
    opt.step()
    delta_rms = (before - p.detach()).pow(2).mean().sqrt().item()
    eps1 = opt.param_groups[0]["eps"][1]
    cap = opt.param_groups[0]["max_relative_update"] * max(eps1, param_rms)
    assert delta_rms == pytest.approx(cap, rel=1e-5)


def _plain_opt(param):
    return Adafactor(
        [param],
        lr=1e-2,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
        stochastic_accumulation=False,
    )


def test_rms_float_roundtrip_and_legacy_tensor_checkpoint():
    """step() stores RMS as a float; a checkpoint that still holds a tensor RMS still steps."""
    p = torch.nn.Parameter(torch.ones(4))
    opt = _plain_opt(p)
    p.grad = torch.ones_like(p)
    opt.step()
    assert isinstance(opt.state[p]["RMS"], float)
    assert math.isfinite(opt.state[p]["RMS"])

    q = torch.nn.Parameter(p.detach().clone())
    opt_q = _plain_opt(q)
    opt_q.load_state_dict(opt.state_dict())
    assert isinstance(opt_q.state[q]["RMS"], float)
    q.grad = torch.ones_like(q)
    opt_q.step()
    assert isinstance(opt_q.state[q]["RMS"], float)
    assert math.isfinite(opt_q.state[q]["RMS"])

    legacy = opt.state_dict()
    legacy["state"][0]["RMS"] = torch.tensor(float("nan"))
    r = torch.nn.Parameter(torch.ones(4))
    opt_r = _plain_opt(r)
    opt_r.load_state_dict(legacy)
    assert torch.is_tensor(opt_r.state[r]["RMS"])
    r.grad = torch.ones_like(r)
    opt_r.step()
    assert isinstance(opt_r.state[r]["RMS"], float)
    assert math.isfinite(opt_r.state[r]["RMS"])
    assert math.isfinite(opt_r.param_groups[0]["rms_max"].item())


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
@pytest.mark.parametrize("beta1", [None, 0.9])
def test_factored_step_stays_finite(shape, beta1):
    p = torch.nn.Parameter(torch.randn(*shape))
    opt = Adafactor(
        [p],
        lr=1e-2,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=beta1,
        weight_decay=0.0,
        factored=True,
        stochastic_accumulation=False,
    )
    p.grad = torch.randn_like(p)
    opt.step()
    assert torch.isfinite(p).all()
    if beta1:
        assert torch.isfinite(opt.state[p]["exp_avg"]).all()
    assert torch.isfinite(opt.state[p]["exp_avg_sq_row"]).all()
    assert torch.isfinite(opt.state[p]["exp_avg_sq_col"]).all()


def test_relative_step_with_scale_parameter_steps():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = Adafactor(
        [p],
        lr=1.0,
        relative_step=True,
        scale_parameter=True,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
        stochastic_accumulation=False,
        stochastic_rounding=False,
    )
    before = p.detach().clone()
    param_rms = before.pow(2).mean().sqrt().item()
    p.grad = torch.randn_like(p)
    opt.step()
    delta_rms = (before - p.detach()).pow(2).mean().sqrt().item()
    eps1 = opt.param_groups[0]["eps"][1]
    cap = opt.param_groups[0]["max_relative_update"] * max(eps1, param_rms)
    assert torch.isfinite(p).all()
    assert delta_rms == pytest.approx(cap, rel=1e-5)
    assert math.isfinite(opt.param_groups[0]["rms_max"].item())
    assert math.isfinite(opt.param_groups[0]["rms_min"].item())


def test_parameter_swapping_skips_params_without_grad():
    params = [torch.nn.Parameter(torch.ones(8)) for _ in range(4)]
    opt = Adafactor(
        params,
        lr=1e-2,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
        do_parameter_swapping=True,
        parameter_swapping_factor=0.25,
        stochastic_accumulation=False,
    )
    assert any(not p.requires_grad for p in params)
    for p in params:
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    opt.step()
    assert all(torch.isfinite(p).all() for p in params)


def test_step_survives_factored_flag_flip_both_ways():
    """Flipping group factored mid-run must recreate buffers without KeyError."""
    p = torch.nn.Parameter(torch.ones(4, 4))
    opt = Adafactor(
        [p],
        lr=1e-3,
        relative_step=False,
        scale_parameter=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
    )
    p.grad = torch.ones_like(p)
    opt.step()
    assert "exp_avg_sq" in opt.state[p]

    opt.param_groups[0]["factored"] = True
    p.grad = torch.ones_like(p)
    opt.step()
    assert "exp_avg_sq_row" in opt.state[p]
    assert "exp_avg_sq_col" in opt.state[p]
    assert "exp_avg_sq" not in opt.state[p]

    opt.param_groups[0]["factored"] = False
    p.grad = torch.ones_like(p)
    opt.step()
    assert "exp_avg_sq" in opt.state[p]
    assert "exp_avg_sq_row" not in opt.state[p]
    assert "exp_avg_sq_col" not in opt.state[p]


def test_set_lr_preserves_per_group_ratios():
    """Runtime set_lr scales all groups; relative ratios stay the same."""
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    opt = Adafactor(
        [
            {"params": [p1], "lr": 1e-4},
            {"params": [p2], "lr": 5e-5},
        ],
        lr=1e-4,
        relative_step=False,
        scale_parameter=False,
        beta1=None,
        weight_decay=0.0,
    )
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert opt.param_groups[1]["lr"] == pytest.approx(5e-5)
    ratio_before = opt.param_groups[0]["lr"] / opt.param_groups[1]["lr"]

    opt.set_lr(2e-4)

    assert opt._lr == pytest.approx(2e-4)
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-4)
    assert opt.param_groups[0]["lr"] / opt.param_groups[1]["lr"] == pytest.approx(ratio_before)


def test_rms_max_decay_rate_rejects_invalid_values():
    p = torch.nn.Parameter(torch.ones(2))
    for bad in (0.0, -0.1, 1.01):
        with pytest.raises(ValueError, match="rms_max_decay_rate"):
            Adafactor(
                [p],
                lr=1e-3,
                rms_max_decay_rate=bad,
                relative_step=False,
                scale_parameter=False,
                beta1=None,
                weight_decay=0.0,
            )


def test_stochastic_accum_hooks_do_not_stack_on_rebuild():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    opt1 = Adafactor(
        [p],
        lr=1e-3,
        stochastic_accumulation=True,
        relative_step=False,
        scale_parameter=False,
        beta1=None,
        weight_decay=0.0,
    )
    handle1 = getattr(p, "_adafactor_stoch_hook", None)
    assert handle1 is not None
    opt2 = Adafactor(
        [p],
        lr=1e-3,
        stochastic_accumulation=True,
        relative_step=False,
        scale_parameter=False,
        beta1=None,
        weight_decay=0.0,
    )
    handle2 = getattr(p, "_adafactor_stoch_hook", None)
    assert handle2 is not None
    assert handle2 is not handle1
    # Prior handle must have been removed (idempotent remove is fine; identity changed).
    assert opt1 is not opt2


def test_cpu_bf16_stochastic_rounding_falls_back_without_assert():
    """CPU non-fp32 + stochastic_rounding must use update_parameter (no copy_stochastic assert)."""
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = Adafactor(
        [p],
        lr=1e-2,
        stochastic_rounding=True,
        stochastic_accumulation=False,
        relative_step=False,
        scale_parameter=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
    )
    before = p.detach().float().clone()
    p.grad = torch.ones_like(p, dtype=torch.bfloat16)
    opt.step()
    assert not torch.allclose(p.detach().float(), before)
    assert p.dtype == torch.bfloat16


def _adafactor_kwargs(**extra):
    return dict(
        lr=1e-2,
        relative_step=False,
        scale_parameter=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
        **extra,
    )


def test_bf16_param_registers_stoch_hook_and_stays_bf16():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = Adafactor(
        [p],
        stochastic_accumulation=True,
        stochastic_rounding=True,
        **_adafactor_kwargs(),
    )
    assert getattr(p, "_adafactor_stoch_hook", None) is not None
    p.grad = torch.ones_like(p)
    opt.step()
    assert p.dtype == torch.bfloat16
    assert opt.param_groups[0].get("rms_ema") is None or (
        opt.param_groups[0]["rms_ema"].dtype == torch.bfloat16
    )


def test_fp32_param_skips_stochastic_accum_hook():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    Adafactor(
        [p],
        stochastic_accumulation=True,
        stochastic_rounding=True,
        **_adafactor_kwargs(),
    )
    assert getattr(p, "_adafactor_stoch_hook", None) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_cuda_stochastic_rounding_uses_copy_stochastic(monkeypatch):
    from toolkit.optimizers.optimizer_utils import copy_stochastic as orig_copy

    device = torch.device("cuda")
    p = torch.nn.Parameter(torch.ones(4, device=device, dtype=torch.bfloat16))
    calls = {"n": 0}

    def _spy(target, source, eps=None):
        calls["n"] += 1
        orig_copy(target, source, eps)

    monkeypatch.setattr("toolkit.optimizers.adafactor.copy_stochastic", _spy)
    opt = Adafactor(
        [p],
        stochastic_accumulation=False,
        stochastic_rounding=True,
        **_adafactor_kwargs(),
    )
    p.grad = torch.ones_like(p)
    opt.step()
    assert calls["n"] == 1
    assert p.dtype == torch.bfloat16


def _assert_fp32_copy_not_aliased(target, source, before):
    """Write-back source is a distinct fp32 tensor; target is still pre-step."""
    assert source.dtype == torch.float32
    assert source.data_ptr() != target.data_ptr()
    assert torch.equal(target.detach(), before)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_cpu_writeback_uses_unaliased_fp32_copy(dtype, monkeypatch):
    """CPU bf16/fp16 step must write back through update_parameter from a fresh fp32 copy."""
    from toolkit.optimizers.optimizer_utils import update_parameter as orig_update

    p = torch.nn.Parameter(torch.ones(4, dtype=dtype))
    before = p.detach().clone()
    calls = {"update": 0, "copy": 0}

    def _update_spy(target, result_float):
        calls["update"] += 1
        _assert_fp32_copy_not_aliased(target, result_float, before)
        orig_update(target, result_float)

    def _copy_spy(target, source, eps=None):
        calls["copy"] += 1

    monkeypatch.setattr("toolkit.optimizers.adafactor.update_parameter", _update_spy)
    monkeypatch.setattr("toolkit.optimizers.adafactor.copy_stochastic", _copy_spy)
    opt = Adafactor(
        [p],
        stochastic_accumulation=False,
        stochastic_rounding=True,
        **_adafactor_kwargs(),
    )
    grad = torch.ones_like(p)
    p.grad = grad
    grad_before = grad.detach().clone()
    grad_ptr = grad.data_ptr()
    opt.step()
    assert calls == {"update": 1, "copy": 0}
    assert p.dtype == dtype
    assert not torch.equal(p.detach(), before)
    assert p.grad.data_ptr() == grad_ptr
    assert torch.equal(p.grad, grad_before)


def test_fp32_param_updates_in_place_without_writeback(monkeypatch):
    """fp32 parameters are updated directly; neither write-back helper runs."""
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    before = p.detach().clone()
    calls = {"update": 0, "copy": 0}

    def _update_spy(target, result_float):
        calls["update"] += 1

    def _copy_spy(target, source, eps=None):
        calls["copy"] += 1

    monkeypatch.setattr("toolkit.optimizers.adafactor.update_parameter", _update_spy)
    monkeypatch.setattr("toolkit.optimizers.adafactor.copy_stochastic", _copy_spy)
    opt = Adafactor(
        [p],
        stochastic_accumulation=False,
        stochastic_rounding=True,
        **_adafactor_kwargs(),
    )
    grad = torch.ones_like(p)
    p.grad = grad
    grad_before = grad.detach().clone()
    grad_ptr = grad.data_ptr()
    opt.step()
    assert calls == {"update": 0, "copy": 0}
    assert p.dtype == torch.float32
    assert not torch.equal(p.detach(), before)
    assert p.grad.data_ptr() == grad_ptr
    assert torch.equal(p.grad, grad_before)


def test_quantized_param_writeback_uses_unaliased_fp32_copy(monkeypatch):
    """QBytesTensor is updated from a dequantized fp32 copy, not in place."""
    pytest.importorskip("optimum.quanto")
    from optimum.quanto import QBytesTensor, qint8, quantize_weight
    from toolkit.optimizers.optimizer_utils import update_parameter as orig_update

    weight = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [0.25, -0.5, 1.5, -1.0]],
        dtype=torch.float32,
    )
    scale = (weight.abs().amax(dim=1) / 127.0).reshape(-1, 1)
    p = quantize_weight(weight, qint8, axis=0, scale=scale, optimized=False)
    assert isinstance(p, QBytesTensor)
    p.requires_grad_(True)
    before = p.dequantize().detach().clone()
    calls = {"update": 0, "copy": 0}

    def _update_spy(target, result_float):
        calls["update"] += 1
        assert isinstance(target, QBytesTensor)
        assert result_float.dtype == torch.float32
        assert result_float.data_ptr() != target.data_ptr()
        assert result_float.data_ptr() != target._data.data_ptr()
        assert torch.equal(target.dequantize().detach(), before)
        orig_update(target, result_float)

    def _copy_spy(target, source, eps=None):
        calls["copy"] += 1

    monkeypatch.setattr("toolkit.optimizers.adafactor.update_parameter", _update_spy)
    monkeypatch.setattr("toolkit.optimizers.adafactor.copy_stochastic", _copy_spy)
    opt = Adafactor(
        [p],
        lr=1.0,
        stochastic_accumulation=False,
        stochastic_rounding=False,
        relative_step=False,
        scale_parameter=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
    )
    p.grad = torch.ones_like(before)
    opt.step()
    assert calls == {"update": 1, "copy": 0}
    assert isinstance(p, QBytesTensor)
    assert not torch.equal(p.dequantize().detach(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_cuda_writeback_uses_unaliased_fp32_copy(dtype, monkeypatch):
    """CUDA bf16/fp16 step must write back through copy_stochastic from a fresh fp32 copy."""
    from toolkit.optimizers.optimizer_utils import copy_stochastic as orig_copy

    device = torch.device("cuda")
    p = torch.nn.Parameter(torch.ones(4, device=device, dtype=dtype))
    before = p.detach().clone()
    calls = {"update": 0, "copy": 0}

    def _copy_spy(target, source, eps=None):
        calls["copy"] += 1
        _assert_fp32_copy_not_aliased(target, source, before)
        orig_copy(target, source, eps)

    def _update_spy(target, result_float):
        calls["update"] += 1

    monkeypatch.setattr("toolkit.optimizers.adafactor.copy_stochastic", _copy_spy)
    monkeypatch.setattr("toolkit.optimizers.adafactor.update_parameter", _update_spy)
    opt = Adafactor(
        [p],
        stochastic_accumulation=False,
        stochastic_rounding=True,
        **_adafactor_kwargs(),
    )
    p.grad = torch.ones_like(p)
    opt.step()
    assert calls == {"update": 0, "copy": 1}
    assert p.dtype == dtype
    assert not torch.equal(p.detach(), before)

