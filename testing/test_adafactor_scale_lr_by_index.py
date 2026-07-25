"""Tests for Adafactor scale_lr_by_index mode."""

import pytest
import torch

from toolkit.optimizers.adafactor import Adafactor


def _state_with_rms(opt, group_idx=0, param_idx=0, rms=1.0):
    p = opt.param_groups[group_idx]["params"][param_idx]
    state = opt.state[p]
    state["RMS"] = torch.tensor(rms)
    return state


def test_scale_lr_by_index_formula():
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    lr = 1e-3
    eps = (1e-30, 1e-3)
    opt = Adafactor(
        [
            {"params": [p0], "index": 0},
            {"params": [p1], "index": 1},
            {"params": [p2], "index": 2},
        ],
        lr=lr,
        eps=eps,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        scale_lr_by_index=True,
    )
    assert opt._max_index == 2

    for idx, expected_factor in ((0, 1.0), (1, 0.5), (2, 0.0)):
        state = _state_with_rms(opt, group_idx=idx)
        got = opt._get_lr(opt.param_groups[idx], state)
        expected = lr * expected_factor + eps[0]
        assert got == pytest.approx(expected)


def test_scale_lr_by_index_skips_unindexed_with_valid_max():
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    lr = 1e-3
    eps = (1e-30, 1e-3)
    opt = Adafactor(
        [
            {"params": [p0], "index": 0},
            {"params": [p1]},  # no index — skip scaling
            {"params": [p2], "index": 2},
        ],
        lr=lr,
        eps=eps,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        scale_lr_by_index=True,
    )
    assert opt._max_index == 2

    state0 = _state_with_rms(opt, 0)
    assert opt._get_lr(opt.param_groups[0], state0) == pytest.approx(lr + eps[0])

    state1 = _state_with_rms(opt, 1)
    assert opt._get_lr(opt.param_groups[1], state1) == pytest.approx(lr)

    state2 = _state_with_rms(opt, 2)
    assert opt._get_lr(opt.param_groups[2], state2) == pytest.approx(eps[0])


def test_scale_lr_by_index_errors_without_indices():
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="cannot determine max_index"):
        Adafactor(
            [{"params": [p]}],
            lr=1e-3,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            beta1=None,
            weight_decay=0.0,
            scale_lr_by_index=True,
        )


def test_scale_lr_by_index_errors_when_max_index_zero():
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="max_index > 0"):
        Adafactor(
            [{"params": [p], "index": 0}],
            lr=1e-3,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            beta1=None,
            weight_decay=0.0,
            scale_lr_by_index=True,
        )


def test_scale_lr_by_index_disabled_by_default():
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    lr = 1e-3
    opt = Adafactor(
        [
            {"params": [p0], "index": 0},
            {"params": [p1], "index": 1},
        ],
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
    )
    assert opt.scale_lr_by_index is False
    assert opt._max_index is None
    state = _state_with_rms(opt, 1)
    assert opt._get_lr(opt.param_groups[1], state) == pytest.approx(lr)


def test_set_lr_still_scaled_dynamically():
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    eps = (1e-30, 1e-3)
    opt = Adafactor(
        [
            {"params": [p0], "index": 0},
            {"params": [p1], "index": 1},
        ],
        lr=1e-3,
        eps=eps,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        scale_lr_by_index=True,
    )
    opt.set_lr(2e-3)
    state0 = _state_with_rms(opt, 0)
    state1 = _state_with_rms(opt, 1)
    assert opt._get_lr(opt.param_groups[0], state0) == pytest.approx(2e-3 + eps[0])
    assert opt._get_lr(opt.param_groups[1], state1) == pytest.approx(eps[0])


def test_scale_lr_factor_power_curve():
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    lr = 1e-3
    eps = (1e-30, 1e-3)
    opt = Adafactor(
        [
            {"params": [p0], "index": 0},
            {"params": [p1], "index": 1},
            {"params": [p2], "index": 2},
        ],
        lr=lr,
        eps=eps,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        scale_lr_by_index=True,
        scale_lr_factor=2.0,
    )
    assert opt.scale_lr_factor == 2.0
    assert opt._max_index == 2

    for idx, expected_factor in ((0, 1.0), (1, 0.25), (2, 0.0)):
        state = _state_with_rms(opt, group_idx=idx)
        got = opt._get_lr(opt.param_groups[idx], state)
        expected = lr * expected_factor + eps[0]
        assert got == pytest.approx(expected)


def test_scale_lr_factor_errors_when_non_positive():
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="scale_lr_factor must be > 0"):
        Adafactor(
            [{"params": [p]}],
            lr=1e-3,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            beta1=None,
            weight_decay=0.0,
            scale_lr_by_index=False,
            scale_lr_factor=0.0,
        )


def test_scale_lr_factor_ignored_when_mode_off():
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    lr = 1e-3
    opt = Adafactor(
        [
            {"params": [p0], "index": 0},
            {"params": [p1], "index": 1},
        ],
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        scale_lr_by_index=False,
        scale_lr_factor=2.0,
    )
    assert opt.scale_lr_factor == 2.0
    assert opt.scale_lr_by_index is False
    state = _state_with_rms(opt, 1)
    assert opt._get_lr(opt.param_groups[1], state) == pytest.approx(lr)


def _make_indexed_wd_opt(
    weight_decay_mode: str,
    *,
    scale_lr_by_index: bool = True,
    lr: float = 0.2,
    wd: float = 0.1,
    include_unindexed: bool = False,
):
    p0 = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    p1 = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    p2 = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    groups = [
        {"params": [p0], "index": 0},
        {"params": [p1], "index": 1},
        {"params": [p2], "index": 2},
    ]
    if include_unindexed:
        p_u = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
        groups = [
            {"params": [p0], "index": 0},
            {"params": [p_u]},
            {"params": [p2], "index": 2},
        ]
    opt = Adafactor(
        groups,
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=wd,
        weight_decay_mode=weight_decay_mode,
        scale_lr_by_index=scale_lr_by_index,
    )
    return opt


def _zero_grad_step(opt):
    for group in opt.param_groups:
        for p in group["params"]:
            p.grad = torch.zeros_like(p)
    opt.step()


def test_scale_wd_by_index_param_rms():
    opt = _make_indexed_wd_opt("param_rms", lr=0.2, wd=0.1)
    _zero_grad_step(opt)
    effective = opt.get_effective_wd()
    # wd' = wd * (1 - index/max_index); max_index=2
    assert effective[0] == pytest.approx(0.1)
    assert effective[1] == pytest.approx(0.05)
    assert effective[2] == pytest.approx(0.0)
    assert all(g["weight_decay"] == pytest.approx(0.1) for g in opt.param_groups)


def test_scale_wd_by_index_constant():
    opt = _make_indexed_wd_opt("constant", lr=0.2, wd=0.1)
    _zero_grad_step(opt)
    effective = opt.get_effective_wd()
    # wd' = wd * (1 - index/max_index); no * lr
    assert effective[0] == pytest.approx(0.1)
    assert effective[1] == pytest.approx(0.05)
    assert effective[2] == pytest.approx(0.0)
    assert all(g["weight_decay"] == pytest.approx(0.1) for g in opt.param_groups)


def test_scale_wd_by_index_absolute():
    lr = 0.2
    wd = 0.1
    eps = (1e-30, 1e-3)
    opt = Adafactor(
        [
            {"params": [torch.nn.Parameter(torch.ones(4))], "index": 0},
            {"params": [torch.nn.Parameter(torch.ones(4))], "index": 1},
            {"params": [torch.nn.Parameter(torch.ones(4))], "index": 2},
        ],
        lr=lr,
        eps=eps,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=wd,
        weight_decay_mode="absolute",
        scale_lr_by_index=True,
    )
    _zero_grad_step(opt)
    # lr_scaled: index 0 -> lr+eps0, index 1 -> 0.5*lr+eps0, index 2 -> eps0
    # wd' = wd * (1 - index/max_index); max_index=2
    wd_scaled = [0.1, 0.05, 0.0]
    expected = [
        wd_scaled[0] * (lr + eps[0]),
        wd_scaled[1] * (0.5 * lr + eps[0]),
        wd_scaled[2] * (eps[0]),
    ]
    effective = opt.get_effective_wd()
    for got, exp in zip(effective, expected):
        assert got == pytest.approx(exp)


def test_scale_wd_by_index_update_rms_zero_grad():
    opt = _make_indexed_wd_opt("update_rms", lr=0.2, wd=0.1)
    _zero_grad_step(opt)
    assert all(v == pytest.approx(0.0) for v in opt.get_effective_wd())


def test_scale_wd_skips_unindexed_group():
    opt = _make_indexed_wd_opt(
        "param_rms", lr=0.2, wd=0.1, include_unindexed=True
    )
    _zero_grad_step(opt)
    effective = opt.get_effective_wd()
    # index 0 -> 0.1; unindexed -> wd * rms; index 2 -> 0.0
    assert effective[0] == pytest.approx(0.1)
    assert effective[1] == pytest.approx(0.1)
    assert effective[2] == pytest.approx(0.0)


def test_scale_wd_disabled_when_mode_off():
    opt = _make_indexed_wd_opt(
        "param_rms", scale_lr_by_index=False, lr=0.2, wd=0.1
    )
    _zero_grad_step(opt)
    assert all(v == pytest.approx(0.1) for v in opt.get_effective_wd())
