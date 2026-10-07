"""scale_parameter caps applied update RMS; it does not scale the learning rate."""
import math

import pytest
import torch

from toolkit.optimizers.adafactor import Adafactor


def _rms(tensor):
    return tensor.detach().float().pow(2).mean().sqrt().item()


def _make(params, **kwargs):
    if isinstance(params, torch.nn.Parameter):
        params = [params]
    defaults = dict(
        lr=1e-3,
        relative_step=False,
        scale_parameter=True,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
        stochastic_accumulation=False,
        stochastic_rounding=False,
    )
    defaults.update(kwargs)
    return Adafactor(params, **defaults)


def _step_delta_rms(opt, param, grad):
    before = param.detach().clone()
    param_rms = _rms(before)
    param.grad = grad
    opt.step()
    return param_rms, _rms(before - param.detach())


def test_default_max_relative_update():
    param = torch.nn.Parameter(torch.zeros(2))
    opt = _make(param, scale_parameter=False)
    assert opt.param_groups[0]["max_relative_update"] == pytest.approx(0.1)
    assert opt._max_relative_update == pytest.approx(0.1)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_max_relative_update_rejects_non_positive(bad):
    param = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="max_relative_update"):
        _make(param, max_relative_update=bad)


def test_cap_binds_at_relative_limit():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(64, 64))
    opt = _make(param, lr=1.0)
    grad = torch.randn_like(param)
    param_rms, delta_rms = _step_delta_rms(opt, param, grad)
    cap = 0.1 * max(1e-3, param_rms)
    assert delta_rms == pytest.approx(cap, rel=1e-5)


def test_cap_does_not_bind_matches_unscaled_lr():
    """Below the cap, scale_parameter=True matches scale_parameter=False at the same lr."""
    torch.manual_seed(0)
    base = torch.randn(64, 64)
    grad = torch.randn(64, 64)
    capped = torch.nn.Parameter(base.clone())
    plain = torch.nn.Parameter(base.clone())
    opt_capped = _make(capped, lr=1e-6, scale_parameter=True)
    opt_plain = _make(plain, lr=1e-6, scale_parameter=False)
    capped.grad = grad.clone()
    plain.grad = grad.clone()
    opt_capped.step()
    opt_plain.step()
    assert torch.equal(capped, plain)


def test_disabled_scale_parameter_ignores_max_relative_update():
    torch.manual_seed(0)
    base = torch.randn(32, 32)
    grad = torch.randn_like(base)
    loose = torch.nn.Parameter(base.clone())
    tight = torch.nn.Parameter(base.clone())
    opt_loose = _make(
        loose, lr=1.0, scale_parameter=False, max_relative_update=0.5
    )
    opt_tight = _make(
        tight, lr=1.0, scale_parameter=False, max_relative_update=0.01
    )
    loose.grad = grad.clone()
    tight.grad = grad.clone()
    opt_loose.step()
    opt_tight.step()
    assert torch.equal(loose, tight)
    assert _rms(base - loose) > 0.5


def test_custom_max_relative_update_is_tighter_than_default():
    torch.manual_seed(0)
    base = torch.randn(32, 32)
    grad = torch.randn_like(base)
    param_rms = _rms(base)
    wide = torch.nn.Parameter(base.clone())
    narrow = torch.nn.Parameter(base.clone())
    opt_wide = _make(wide, lr=1.0, max_relative_update=0.1)
    opt_narrow = _make(narrow, lr=1.0, max_relative_update=0.01)
    wide.grad = grad.clone()
    narrow.grad = grad.clone()
    opt_wide.step()
    opt_narrow.step()
    wide_delta = _rms(base - wide)
    narrow_delta = _rms(base - narrow)
    assert wide_delta == pytest.approx(0.1 * param_rms, rel=1e-5)
    assert narrow_delta == pytest.approx(0.01 * param_rms, rel=1e-5)
    assert narrow_delta < wide_delta


def test_zero_init_param_still_moves_up_to_eps_floor():
    param = torch.nn.Parameter(torch.zeros(32, 32))
    opt = _make(param, lr=1.0)
    eps1 = opt.param_groups[0]["eps"][1]
    param.grad = torch.ones_like(param)
    opt.step()
    delta_rms = _rms(param)
    cap = 0.1 * eps1
    assert delta_rms > 0.0
    assert delta_rms == pytest.approx(cap, rel=1e-4)


def test_momentum_buffer_is_not_rescaled_by_the_cap():
    torch.manual_seed(0)
    base = torch.randn(16, 16)
    grads = [torch.randn(16, 16) for _ in range(5)]
    capped = torch.nn.Parameter(base.clone())
    free = torch.nn.Parameter(base.clone())
    opt_capped = _make(capped, lr=1.0, beta1=0.9, max_relative_update=0.1)
    opt_free = _make(free, lr=1.0, beta1=0.9, max_relative_update=1e6)
    for grad in grads:
        capped.grad = grad.clone()
        free.grad = grad.clone()
        opt_capped.step()
        opt_free.step()
    assert torch.equal(opt_capped.state[capped]["exp_avg"], opt_free.state[free]["exp_avg"])
    assert not torch.equal(capped, free)


def test_update_rms_weight_decay_uses_post_cap_rms():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(32, 32))
    weight_decay = 0.05
    opt = _make(
        param,
        lr=1.0,
        weight_decay=weight_decay,
        weight_decay_mode="update_rms",
    )
    before = param.detach().clone()
    param_rms = _rms(before)
    param.grad = torch.randn_like(param)
    opt.step()
    cap = 0.1 * max(1e-3, param_rms)
    assert opt.state[param]["effective_wd"] == pytest.approx(weight_decay * cap, rel=1e-4)


def test_absolute_weight_decay_uses_plain_lr():
    param = torch.nn.Parameter(torch.full((16, 16), 2.0))
    lr = 1e-2
    weight_decay = 0.1
    opt = _make(
        param,
        lr=lr,
        scale_parameter=True,
        weight_decay=weight_decay,
        weight_decay_mode="absolute",
    )
    param.grad = torch.randn_like(param)
    opt.step()
    # Absolute weight decay is weight_decay * lr. Parameter RMS (2) is not a factor.
    assert opt.state[param]["effective_wd"] == pytest.approx(weight_decay * lr)


def test_bf16_written_step_respects_cap():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    param = torch.nn.Parameter(
        torch.randn(32, 32, device=device, dtype=torch.bfloat16)
    )
    opt = _make(param, lr=1.0, stochastic_rounding=True)
    before = param.detach().float().clone()
    param_rms = _rms(before)
    param.grad = torch.randn(param.shape, device=device, dtype=torch.bfloat16)
    opt.step()
    delta_rms = _rms(before - param.detach().float())
    eps1 = opt.param_groups[0]["eps"][1]
    cap = 0.1 * max(eps1, param_rms)
    # bf16 rounding of the parameter can add about one ulp of RMS.
    rounding = param_rms * (2 ** -6)
    assert torch.isfinite(param).all()
    assert 0.0 < delta_rms <= cap + rounding


def test_warmup_and_index_scaling_respect_cap():
    torch.manual_seed(0)
    groups = [
        {
            "params": [torch.nn.Parameter(torch.randn(8, 8))],
            "index": i,
            "name": f"layers_{i}",
        }
        for i in range(4)
    ]
    opt = Adafactor(
        groups,
        lr=1.0,
        scale_parameter=True,
        max_relative_update=0.1,
        warmup_init=True,
        warmup_steps=5,
        scale_lr_by_index=True,
        scale_lr_mean=1.0,
        scale_lr_std=0.5,
        scale_lr_mask=["layers"],
        relative_step=False,
        beta1=None,
        weight_decay=0.0,
        factored=False,
        stochastic_accumulation=False,
        stochastic_rounding=False,
    )
    eps1 = opt.param_groups[0]["eps"][1]
    for _ in range(8):
        snapshots = []
        for group in opt.param_groups:
            for param in group["params"]:
                snapshots.append((param, param.detach().clone()))
                param.grad = torch.randn_like(param)
        opt.step()
        for param, before in snapshots:
            delta_rms = _rms(before - param.detach())
            cap = 0.1 * max(eps1, _rms(before))
            assert delta_rms <= cap * (1.0 + 1e-4)
            assert torch.isfinite(param).all()


def test_checkpoint_keeps_current_run_max_relative_update():
    torch.manual_seed(0)
    source_param = torch.nn.Parameter(torch.randn(8, 8))
    source = _make(source_param, lr=1.0, max_relative_update=0.2)
    source_param.grad = torch.randn_like(source_param)
    source.step()
    state = source.state_dict()

    dest_param = torch.nn.Parameter(torch.randn(8, 8))
    dest = _make(dest_param, lr=1.0, max_relative_update=0.05)
    dest.load_state_dict(state)
    assert dest.param_groups[0]["max_relative_update"] == pytest.approx(0.05)

    before = dest_param.detach().clone()
    param_rms = _rms(before)
    dest_param.grad = torch.randn_like(dest_param)
    dest.step()
    delta_rms = _rms(before - dest_param.detach())
    cap = 0.05 * max(1e-3, param_rms)
    assert delta_rms == pytest.approx(cap, rel=1e-4)


def test_legacy_checkpoint_without_key_uses_default():
    param = torch.nn.Parameter(torch.ones(4))
    opt = _make(param, scale_parameter=True)
    param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state_dict()
    for group in state["param_groups"]:
        group.pop("max_relative_update", None)

    restored_param = torch.nn.Parameter(torch.ones(4))
    restored = _make(restored_param, scale_parameter=True)
    restored.load_state_dict(state)
    assert restored.param_groups[0]["max_relative_update"] == pytest.approx(0.1)


def test_per_group_max_relative_update():
    torch.manual_seed(0)
    base = torch.randn(32, 32)
    grad = torch.randn_like(base)
    wide = torch.nn.Parameter(base.clone())
    narrow = torch.nn.Parameter(base.clone())
    opt = _make(
        [
            {"params": [wide], "max_relative_update": 0.1},
            {"params": [narrow], "max_relative_update": 0.01},
        ],
        lr=1.0,
    )
    param_rms = _rms(base)
    wide.grad = grad.clone()
    narrow.grad = grad.clone()
    opt.step()
    assert _rms(base - wide) == pytest.approx(0.1 * param_rms, rel=1e-5)
    assert _rms(base - narrow) == pytest.approx(0.01 * param_rms, rel=1e-5)


def test_zero_grad_does_not_divide_by_zero():
    param = torch.nn.Parameter(torch.randn(8, 8))
    opt = _make(param, lr=1.0)
    before = param.detach().clone()
    param.grad = torch.zeros_like(param)
    opt.step()
    assert torch.isfinite(param).all()
    assert torch.equal(param, before)


def test_nan_grad_stays_finite_under_cap():
    param = torch.nn.Parameter(torch.randn(8, 8))
    opt = _make(param, lr=1.0)
    param.grad = torch.full_like(param, float("nan"))
    opt.step()
    assert torch.isfinite(param).all()
    assert math.isfinite(opt.state[param]["RMS"])


def test_lr_none_is_bounded_by_cap():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(16, 16))
    opt = _make(param, lr=None)
    before = param.detach().clone()
    param_rms = _rms(before)
    param.grad = torch.randn_like(param)
    opt.step()
    cap = 0.1 * max(1e-3, param_rms)
    assert opt._get_lr(opt.param_groups[0]) == pytest.approx(1.0)
    assert _rms(before - param.detach()) == pytest.approx(cap, rel=1e-5)


def test_cap_floor_does_not_collapse_when_param_rms_shrinks():
    param = torch.nn.Parameter(torch.full((64,), 1.0))
    opt = _make(param, lr=1.0, max_relative_update=0.1)
    eps1 = opt.param_groups[0]["eps"][1]
    saw_below_eps = False
    for _ in range(80):
        before = param.detach().clone()
        param_rms = _rms(before)
        param.grad = torch.ones_like(param)
        opt.step()
        delta_rms = _rms(before - param.detach())
        cap = 0.1 * max(eps1, param_rms)
        assert delta_rms == pytest.approx(cap, rel=1e-4)
        assert torch.isfinite(param).all()
        if param_rms < eps1:
            saw_below_eps = True
            assert cap == pytest.approx(0.1 * eps1)
            assert delta_rms > 0.0
    assert saw_below_eps
