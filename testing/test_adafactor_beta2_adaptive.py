"""Tests for Adafactor adaptive beta2 behavior on stale second-moment regimes."""

import pytest
import torch

from toolkit.optimizers.adafactor import Adafactor


def _normalized_direction(shape):
    base = torch.linspace(1.0, float(shape[0] * shape[1]), steps=shape[0] * shape[1])
    direction = base.view(*shape)
    return direction / direction.norm()


def _run_warm_then_tiny(beta2_adaptive: bool):
    shape = (8, 16)
    direction = _normalized_direction(shape)
    p = torch.nn.Parameter(torch.full(shape, 0.01))
    opt = Adafactor(
        [p],
        lr=1e-4,
        beta1=None,
        beta2=0.999,
        beta2_adaptive=beta2_adaptive,
        beta2_min=0.9,
        scale_parameter=False,
        relative_step=False,
        weight_decay=0.0,
        clip_threshold=1.0,
    )

    for _ in range(120):
        p.grad = (direction * 1e-2).clone()
        opt.step()

    before = p.detach().clone()
    for _ in range(20):
        p.grad = (direction * 1e-5).clone()
        opt.step()

    st = opt.state[p]
    update_rms = (before - p.detach()).pow(2).mean().sqrt().item()
    return update_rms, float(st["beta2_effective"]), float(st["exp_avg_sq_row"].mean().item()), opt.get_mean_beta2()


def test_beta2_effective_tracks_config_when_adaptive_disabled():
    _, beta2_eff, _, beta2_mean = _run_warm_then_tiny(beta2_adaptive=False)
    assert beta2_eff == pytest.approx(0.999, rel=1e-6)
    assert beta2_mean == pytest.approx(0.999, rel=1e-6)


def test_adaptive_beta2_recovers_from_stale_second_moment():
    static_update, _, static_v, _ = _run_warm_then_tiny(beta2_adaptive=False)
    adaptive_update, adaptive_beta2, adaptive_v, adaptive_beta2_mean = _run_warm_then_tiny(beta2_adaptive=True)

    assert adaptive_beta2 < 0.95
    assert adaptive_beta2_mean < 0.95
    assert adaptive_v < static_v * 0.2
    assert adaptive_update > static_update * 1.4
