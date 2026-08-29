"""Tests for Adafactor Gaussian scale_lr_by_index mode."""

import math

import pytest
import torch

from extensions_built_in.sd_trainer.gaussian_timestep_weights import (
    evaluate_gaussian_timestep,
)
from toolkit.optimizer import get_optimizer
from toolkit.optimizers.adafactor import Adafactor


def _state_with_rms(opt, group_idx=0, param_idx=0, rms=1.0):
    p = opt.param_groups[group_idx]["params"][param_idx]
    state = opt.state[p]
    state["RMS"] = torch.tensor(rms)
    return state


def _make_gaussian_opt(
    groups,
    *,
    lr=1e-3,
    eps=(1e-30, 1e-3),
    mean=0.0,
    std=0.5,
    mask=None,
    scale_lr_by_index=True,
    weight_decay=0.0,
    weight_decay_mode="absolute",
    **kwargs,
):
    return Adafactor(
        groups,
        lr=lr,
        eps=eps,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=weight_decay,
        weight_decay_mode=weight_decay_mode,
        scale_lr_by_index=scale_lr_by_index,
        scale_lr_mean=mean,
        scale_lr_std=std,
        scale_lr_mask=mask,
        **kwargs,
    )


def test_gaussian_lr_lookup_matches_timestep_helper():
    max_index = 10
    mean = 4.0
    std = 0.25
    ntt = max_index + 1
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(2))], "index": i}
        for i in range(ntt)
    ]
    opt = _make_gaussian_opt(groups, mean=mean, std=std)
    assert opt._max_index == max_index
    assert opt._scale_lr_lookup is not None
    assert opt._scale_lr_lookup.shape == (ntt,)

    slots = torch.arange(ntt, dtype=torch.float32)
    ref = evaluate_gaussian_timestep(
        slots,
        mean,
        std,
        device="cpu",
        dtype=torch.float32,
        num_train_timesteps=ntt,
        gaussian_shift=0.0,
    )
    assert torch.allclose(opt._scale_lr_lookup, ref, rtol=0.0, atol=1e-6)

    lr = 1e-3
    eps0 = 1e-30
    for idx in range(ntt):
        state = _state_with_rms(opt, group_idx=idx)
        got = opt._get_lr(opt.param_groups[idx], state)
        expected = lr * float(ref[idx].item()) + eps0
        assert got == pytest.approx(expected)


def test_gaussian_zero_weight_at_min_after_minmax():
    """Min-max normalization drives the lowest PDF sample to weight 0."""
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(2))], "index": i}
        for i in range(5)
    ]
    opt = _make_gaussian_opt(groups, mean=0.0, std=0.2)
    weights = opt._scale_lr_lookup
    assert float(weights.min().item()) == pytest.approx(0.0)
    assert float(weights.max().item()) == pytest.approx(1.0)
    # Peak near mean=0 => index 0 is 1; far end is 0.
    assert float(weights[0].item()) == pytest.approx(1.0)
    assert float(weights[-1].item()) == pytest.approx(0.0)
    state = _state_with_rms(opt, group_idx=4)
    assert opt._get_lr(opt.param_groups[4], state) == pytest.approx(1e-30)


def test_scale_lr_by_index_skips_unindexed_with_valid_max():
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    lr = 1e-3
    eps = (1e-30, 1e-3)
    opt = _make_gaussian_opt(
        [
            {"params": [p0], "index": 0},
            {"params": [p1]},
            {"params": [p2], "index": 2},
        ],
        lr=lr,
        eps=eps,
        mean=0.0,
        std=0.5,
    )
    assert opt._max_index == 2

    state0 = _state_with_rms(opt, 0)
    w0 = float(opt._scale_lr_lookup[0].item())
    assert opt._get_lr(opt.param_groups[0], state0) == pytest.approx(
        lr * w0 + eps[0]
    )

    state1 = _state_with_rms(opt, 1)
    assert opt._get_lr(opt.param_groups[1], state1) == pytest.approx(lr)

    state2 = _state_with_rms(opt, 2)
    w2 = float(opt._scale_lr_lookup[2].item())
    assert opt._get_lr(opt.param_groups[2], state2) == pytest.approx(
        lr * w2 + eps[0]
    )


def test_scale_lr_by_index_errors_without_indices():
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="cannot determine max_index"):
        _make_gaussian_opt([{"params": [p]}], mean=0.0, std=0.5)


def test_scale_lr_by_index_errors_when_max_index_zero():
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="max_index > 0"):
        _make_gaussian_opt(
            [{"params": [p], "index": 0}], mean=0.0, std=0.5
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
    assert opt.scale_lr_mean is None
    assert opt.scale_lr_std is None
    state = _state_with_rms(opt, 1)
    assert opt._get_lr(opt.param_groups[1], state) == pytest.approx(lr)


def test_validation_requires_mean_std_when_enabled():
    p = torch.nn.Parameter(torch.ones(2))
    groups = [{"params": [p], "index": 0}, {"params": [torch.nn.Parameter(torch.ones(2))], "index": 1}]
    with pytest.raises(ValueError, match="scale_lr_mean"):
        Adafactor(
            groups,
            lr=1e-3,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            beta1=None,
            weight_decay=0.0,
            scale_lr_by_index=True,
        )
    with pytest.raises(ValueError, match="scale_lr_std"):
        Adafactor(
            groups,
            lr=1e-3,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            beta1=None,
            weight_decay=0.0,
            scale_lr_by_index=True,
            scale_lr_mean=1.0,
            scale_lr_std=0.0,
        )
    with pytest.raises(ValueError, match="scale_lr_mean"):
        Adafactor(
            groups,
            lr=1e-3,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            beta1=None,
            weight_decay=0.0,
            scale_lr_by_index=True,
            scale_lr_mean=float("nan"),
            scale_lr_std=0.5,
        )


def test_validation_mask_rejects_empty_and_non_sequence_str():
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    groups = [
        {"params": [p0], "index": 0, "name": "layers_0"},
        {"params": [p1], "index": 1, "name": "layers_1"},
    ]
    with pytest.raises(ValueError, match="scale_lr_mask"):
        _make_gaussian_opt(groups, mask=[""])
    with pytest.raises(ValueError, match="scale_lr_mask"):
        _make_gaussian_opt(groups, mask="layers")


def test_validation_mask_rejects_non_sequence_and_mapping():
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(2))], "index": 0, "name": "a"},
        {"params": [torch.nn.Parameter(torch.ones(2))], "index": 1, "name": "b"},
    ]
    with pytest.raises(ValueError, match="scale_lr_mask"):
        _make_gaussian_opt(groups, mask=123)
    with pytest.raises(ValueError, match="scale_lr_mask"):
        _make_gaussian_opt(groups, mask={"layers": True})
    with pytest.raises(ValueError, match="scale_lr_mask"):
        _make_gaussian_opt(groups, mask={"layers"})


def test_mean_std_optional_when_mode_disabled():
    p = torch.nn.Parameter(torch.ones(2))
    opt = Adafactor(
        [{"params": [p], "index": 0}, {"params": [torch.nn.Parameter(torch.ones(2))], "index": 1}],
        lr=1e-3,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        scale_lr_by_index=False,
    )
    assert opt.scale_lr_mean is None
    assert opt.scale_lr_std is None


def test_set_lr_still_scaled_dynamically():
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    eps = (1e-30, 1e-3)
    opt = _make_gaussian_opt(
        [
            {"params": [p0], "index": 0},
            {"params": [p1], "index": 1},
        ],
        lr=1e-3,
        eps=eps,
        mean=0.0,
        std=0.5,
    )
    opt.set_lr(2e-3)
    state0 = _state_with_rms(opt, 0)
    state1 = _state_with_rms(opt, 1)
    w0 = float(opt._scale_lr_lookup[0].item())
    w1 = float(opt._scale_lr_lookup[1].item())
    assert opt._get_lr(opt.param_groups[0], state0) == pytest.approx(
        2e-3 * w0 + eps[0]
    )
    assert opt._get_lr(opt.param_groups[1], state1) == pytest.approx(
        2e-3 * w1 + eps[0]
    )


def test_mask_none_and_empty_apply_to_all_indexed():
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(2))], "index": 0, "name": "a"},
        {"params": [torch.nn.Parameter(torch.ones(2))], "index": 1, "name": "b"},
    ]
    for mask in (None, []):
        opt = _make_gaussian_opt(groups, mean=0.0, std=0.5, mask=mask)
        assert opt._scale_lr_applies(opt.param_groups[0])
        assert opt._scale_lr_applies(opt.param_groups[1])
        state = _state_with_rms(opt, 1)
        w1 = float(opt._scale_lr_lookup[1].item())
        assert opt._get_lr(opt.param_groups[1], state) == pytest.approx(
            1e-3 * w1 + 1e-30
        )


def test_mask_or_match_case_sensitive():
    groups = [
        {
            "params": [torch.nn.Parameter(torch.ones(2))],
            "index": 0,
            "name": "layers_0",
        },
        {
            "params": [torch.nn.Parameter(torch.ones(2))],
            "index": 1,
            "name": "attn_1",
        },
        {
            "params": [torch.nn.Parameter(torch.ones(2))],
            "index": 2,
            "name": "Layers_2",
        },
    ]
    opt = _make_gaussian_opt(
        groups, mean=1.0, std=0.5, mask=["layers", "attn"]
    )
    assert opt._scale_lr_applies(opt.param_groups[0]) is True
    assert opt._scale_lr_applies(opt.param_groups[1]) is True
    # "Layers" does not match case-sensitive "layers"
    assert opt._scale_lr_applies(opt.param_groups[2]) is False

    lr = 1e-3
    eps0 = 1e-30
    state2 = _state_with_rms(opt, 2)
    assert opt._get_lr(opt.param_groups[2], state2) == pytest.approx(lr)

    state0 = _state_with_rms(opt, 0)
    w0 = float(opt._scale_lr_lookup[0].item())
    assert opt._get_lr(opt.param_groups[0], state0) == pytest.approx(
        lr * w0 + eps0
    )


def test_mask_no_match_keeps_original_lr():
    groups = [
        {
            "params": [torch.nn.Parameter(torch.ones(2))],
            "index": 0,
            "name": "other",
        },
        {
            "params": [torch.nn.Parameter(torch.ones(2))],
            "index": 1,
            "name": "also_other",
        },
    ]
    opt = _make_gaussian_opt(groups, mean=0.0, std=0.5, mask=["layers"])
    assert opt._max_index == 1
    for idx in (0, 1):
        state = _state_with_rms(opt, idx)
        assert opt._get_lr(opt.param_groups[idx], state) == pytest.approx(1e-3)


def test_max_index_uses_all_indexed_before_mask():
    groups = [
        {
            "params": [torch.nn.Parameter(torch.ones(2))],
            "index": 0,
            "name": "layers_0",
        },
        {
            "params": [torch.nn.Parameter(torch.ones(2))],
            "index": 5,
            "name": "skip_me",
        },
    ]
    opt = _make_gaussian_opt(groups, mean=0.0, std=0.5, mask=["layers"])
    assert opt._max_index == 5
    assert opt._scale_lr_lookup.shape == (6,)
    assert opt._scale_lr_applies(opt.param_groups[0]) is True
    assert opt._scale_lr_applies(opt.param_groups[1]) is False


def test_set_scale_lr_config_recomputes_lookup():
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(2))], "index": i}
        for i in range(3)
    ]
    opt = _make_gaussian_opt(groups, mean=0.0, std=0.5)
    before = opt._scale_lr_lookup.clone()
    opt.set_scale_lr_config(mean=2.0, std=0.2, mask=None)
    assert opt.scale_lr_mean == pytest.approx(2.0)
    assert opt.scale_lr_std == pytest.approx(0.2)
    assert not torch.allclose(opt._scale_lr_lookup, before)

    ref = evaluate_gaussian_timestep(
        torch.arange(3, dtype=torch.float32),
        2.0,
        0.2,
        device="cpu",
        dtype=torch.float32,
        num_train_timesteps=3,
    )
    assert torch.allclose(opt._scale_lr_lookup, ref, rtol=0.0, atol=1e-6)


def test_set_scale_lr_by_index_enables_after_init_false():
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(2))], "index": i}
        for i in range(3)
    ]
    opt = _make_gaussian_opt(
        groups, mean=0.0, std=0.5, scale_lr_by_index=False
    )
    assert opt.scale_lr_by_index is False
    assert opt._max_index is None
    assert opt.scale_lr_mean == pytest.approx(0.0)
    assert opt.scale_lr_std == pytest.approx(0.5)

    opt.set_scale_lr_by_index(True)
    assert opt.scale_lr_by_index is True
    assert opt._max_index == 2
    assert opt._scale_lr_lookup is not None

    lr = 1e-3
    eps0 = 1e-30
    for idx in (0, 1, 2):
        state = _state_with_rms(opt, group_idx=idx)
        w = float(opt._scale_lr_lookup[idx].item())
        assert opt._get_lr(opt.param_groups[idx], state) == pytest.approx(
            lr * w + eps0
        )


def test_set_scale_lr_by_index_disables():
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(2))], "index": 0},
        {"params": [torch.nn.Parameter(torch.ones(2))], "index": 1},
    ]
    opt = _make_gaussian_opt(groups, mean=0.0, std=0.5)
    opt.set_scale_lr_by_index(False)
    assert opt.scale_lr_by_index is False
    assert opt._max_index is None
    assert opt._scale_lr_lookup is None
    state = _state_with_rms(opt, 1)
    assert opt._get_lr(opt.param_groups[1], state) == pytest.approx(1e-3)


def test_set_scale_lr_by_index_errors_without_mean_std():
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    opt = Adafactor(
        [{"params": [p0], "index": 0}, {"params": [p1], "index": 1}],
        lr=1e-3,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        scale_lr_by_index=False,
    )
    with pytest.raises(ValueError, match="scale_lr_mean"):
        opt.set_scale_lr_by_index(True)


def test_set_scale_lr_by_index_errors_without_indices():
    p = torch.nn.Parameter(torch.ones(2))
    opt = Adafactor(
        [{"params": [p]}],
        lr=1e-3,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
        scale_lr_by_index=False,
        scale_lr_mean=0.0,
        scale_lr_std=0.5,
    )
    with pytest.raises(ValueError, match="cannot determine max_index"):
        opt.set_scale_lr_by_index(True)


def test_get_optimizer_drops_legacy_scale_lr_keys(capsys):
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    opt = get_optimizer(
        [{"params": [p0], "index": 0}, {"params": [p1], "index": 1}],
        optimizer_type="adafactor",
        learning_rate=1e-3,
        optimizer_params={
            "relative_step": False,
            "scale_parameter": False,
            "warmup_init": False,
            "beta1": None,
            "weight_decay": 0.0,
            "scale_lr_by_index": True,
            "scale_lr_mean": 0.0,
            "scale_lr_std": 0.5,
            "scale_lr_factor": 1.2,
            "weight_decay_max": 0.1,
        },
    )
    assert isinstance(opt, Adafactor)
    assert not hasattr(opt, "scale_lr_factor")
    assert not hasattr(opt, "weight_decay_max")
    captured = capsys.readouterr().out
    assert "scale_lr_factor" in captured
    assert "weight_decay_max" in captured


def _zero_grad_step(opt):
    for group in opt.param_groups:
        for p in group["params"]:
            p.grad = torch.zeros_like(p)
    opt.step()


def test_weight_decay_unchanged_across_indexes_constant():
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(4))], "index": i}
        for i in range(3)
    ]
    opt = _make_gaussian_opt(
        groups,
        lr=0.2,
        mean=0.0,
        std=0.5,
        weight_decay=0.1,
        weight_decay_mode="constant",
    )
    _zero_grad_step(opt)
    effective = opt.get_effective_wd()
    assert all(v == pytest.approx(0.1) for v in effective)
    assert all(g["weight_decay"] == pytest.approx(0.1) for g in opt.param_groups)


def test_weight_decay_absolute_equal_across_gaussian_weights():
    """Absolute WD uses pre-Gaussian LR; equal effective WD even with a zero weight."""
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(4))], "index": i}
        for i in range(5)
    ]
    lr = 0.2
    wd = 0.1
    opt = _make_gaussian_opt(
        groups,
        lr=lr,
        mean=0.0,
        std=0.2,
        weight_decay=wd,
        weight_decay_mode="absolute",
    )
    weights = [float(w.item()) for w in opt._scale_lr_lookup]
    assert weights[0] == pytest.approx(1.0)
    assert weights[-1] == pytest.approx(0.0)
    assert weights[0] != pytest.approx(weights[2])

    _zero_grad_step(opt)
    effective = opt.get_effective_wd()
    expected = wd * lr
    assert all(v == pytest.approx(expected) for v in effective)
    assert all(g["weight_decay"] == pytest.approx(wd) for g in opt.param_groups)

    # Update LRs still differ (including near-zero at the far index).
    lrs = []
    for idx in range(5):
        state = _state_with_rms(opt, group_idx=idx)
        lrs.append(opt._get_lr(opt.param_groups[idx], state))
    assert lrs[0] > lrs[2] > lrs[-1]
    assert lrs[-1] == pytest.approx(1e-30)


def test_weight_decay_unchanged_across_indexes_param_rms():
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(4))], "index": i}
        for i in range(3)
    ]
    opt = _make_gaussian_opt(
        groups,
        lr=0.2,
        mean=0.0,
        std=0.5,
        weight_decay=0.1,
        weight_decay_mode="param_rms",
    )
    _zero_grad_step(opt)
    effective = opt.get_effective_wd()
    assert all(v == pytest.approx(0.1) for v in effective)
    assert all(g["weight_decay"] == pytest.approx(0.1) for g in opt.param_groups)


def test_weight_decay_disabled_scaling_when_mode_off():
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(4))], "index": i}
        for i in range(3)
    ]
    opt = _make_gaussian_opt(
        groups,
        lr=0.2,
        mean=0.0,
        std=0.5,
        scale_lr_by_index=False,
        weight_decay=0.1,
        weight_decay_mode="param_rms",
    )
    _zero_grad_step(opt)
    assert all(v == pytest.approx(0.1) for v in opt.get_effective_wd())
