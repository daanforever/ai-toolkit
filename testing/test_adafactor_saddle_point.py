"""Tests for Adafactor saddle_point_boost (RMS stagnation heuristic)."""

import pytest
import torch
from unittest.mock import patch

from toolkit.optimizers.adafactor import Adafactor


def _minimal_adafactor(**kwargs):
    p = torch.nn.Parameter(torch.ones(2, 2))
    defaults = dict(
        lr=1e-3,
        relative_step=True,
        scale_parameter=True,
        beta1=None,
        weight_decay=0.0,
        saddle_point_step=0.1,
    )
    defaults.update(kwargs)
    return Adafactor([p], **defaults), p


def test_update_saddle_point_boost_rises_when_stagnant_no_cap():
    opt, _ = _minimal_adafactor(saddle_point_step=0.25)
    opt._saddle_point_boost = 1.0
    assert opt._update_saddle_point_boost(True) == pytest.approx(1.25)
    assert opt._update_saddle_point_boost(True) == pytest.approx(1.5)
    assert opt._saddle_point_boost == pytest.approx(1.5)


def test_update_saddle_point_boost_falls_when_not_stagnant_floors_at_one():
    opt, _ = _minimal_adafactor(saddle_point_step=0.3)
    opt._saddle_point_boost = 1.5
    assert opt._update_saddle_point_boost(False) == pytest.approx(1.2)
    # 1.2 - 0.3 = 0.9, clamped up to 1.0
    assert opt._update_saddle_point_boost(False) == pytest.approx(1.0)
    assert opt._update_saddle_point_boost(False) == pytest.approx(1.0)


def test_update_saddle_point_boost_alternation():
    opt, _ = _minimal_adafactor(saddle_point_step=0.1)
    opt._saddle_point_boost = 1.0
    assert opt._update_saddle_point_boost(True) == pytest.approx(1.1)
    assert opt._update_saddle_point_boost(False) == pytest.approx(1.0)
    assert opt._update_saddle_point_boost(True) == pytest.approx(1.1)


def test_detect_saddle_point_updates_instance_boost():
    opt, _ = _minimal_adafactor(saddle_point_step=0.05)
    opt._saddle_point_boost = 1.0
    with patch.object(opt._saddle_point_detector, "check", return_value=(True, 0.0)):
        opt._detect_saddle_point(1.0)
    assert opt._saddle_point_boost == pytest.approx(1.05)


def test_get_lr_uses_saddle_point_boost_when_relative_step():
    opt, p = _minimal_adafactor(
        lr=1.0,
        min_lr=1e-6,
        saddle_point_step=0.0,
    )
    opt._saddle_point_boost = 1.5
    state = {
        "RMS": torch.tensor(1.0),
        "grad_rms": torch.tensor(0.1),
    }
    opt.param_groups[0]["rms_max"] = torch.tensor(1.0)
    lr = opt._get_lr(opt.param_groups[0], state)
    base = 1.0 * max(opt.param_groups[0]["eps"][1], 1.0)
    ratio = max(opt.param_groups[0]["eps"][0], 0.0)
    expected_relative = (1 + 1e-6 * ratio) * 1.5
    assert lr == pytest.approx(base * expected_relative)
    # lr is a base multiplier, not a cap: saddle boost can push effective LR above group lr
    assert lr > opt.param_groups[0]["lr"]


def test_get_lr_ignores_boost_when_not_relative_step():
    opt, p = _minimal_adafactor(relative_step=False, scale_parameter=False, lr=0.5)
    opt._saddle_point_boost = 99.0
    state = {"RMS": torch.tensor(2.0)}
    lr = opt._get_lr(opt.param_groups[0], state)
    assert lr == pytest.approx(0.5)


def test_step_advances_boost_when_detector_stagnant(monkeypatch):
    opt, p = _minimal_adafactor(saddle_point_step=0.2)
    p.grad = torch.zeros_like(p)
    calls = {"n": 0}

    def fake_check(_rms):
        calls["n"] += 1
        return (True, 0.0)

    monkeypatch.setattr(opt._saddle_point_detector, "check", fake_check)
    opt.step()
    assert opt._saddle_point_boost == pytest.approx(1.2)
    assert opt.get_mean_saddle_point_boost() == pytest.approx(1.2)
