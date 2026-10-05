import math

import pytest
import torch

from toolkit.optimizer import get_optimizer
from toolkit.optimizers.adafactor import Adafactor


def _make_opt(beta2, decay_rate=-0.8):
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 6))
    opt = Adafactor(
        [p],
        lr=1e-3,
        beta1=None,
        beta2=beta2,
        eps=(1e-30, 1e-3),
        scale_parameter=False,
        relative_step=False,
        weight_decay=0.0,
        factored=False,
        decay_rate=decay_rate,
        stochastic_accumulation=False,
        stochastic_rounding=False,
    )
    return opt, p


def _run(beta2, mid_set=None, steps=15):
    opt, p = _make_opt(beta2)
    updates = []
    for i in range(steps):
        p.grad = torch.full((6,), 0.05 * ((i % 5) + 1))
        if mid_set is not None and i == 7:
            opt.set_beta2(mid_set)
        before = p.detach().clone()
        opt.step()
        updates.append((before - p.detach()).clone())
    return p.detach().clone(), opt.state[p]["exp_avg_sq"].clone(), opt.get_mean_beta2(), updates


def test_stored_beta2_does_not_affect_parameters_or_second_moment():
    p09, v09, mean09, _ = _run(0.9)
    p99, v99, mean99, _ = _run(0.99)
    assert torch.equal(p09, p99)
    assert torch.equal(v09, v99)
    assert mean09 == pytest.approx(0.9)
    assert mean99 == pytest.approx(0.99)


def test_set_beta2_does_not_change_next_update():
    _, _, _, updates_ctrl = _run(0.9)
    _, _, mean_after, updates_set = _run(0.9, mid_set=0.5)
    assert torch.equal(updates_ctrl[7], updates_set[7])
    assert torch.equal(torch.stack(updates_ctrl), torch.stack(updates_set))
    assert mean_after == pytest.approx(0.5)


def test_second_moment_matches_time_decay_recurrence():
    """v = beta2t * v + (1 - beta2t) * (grad^2 + eps), beta2t = 1 - step ** decay_rate."""
    torch.manual_seed(0)
    eps0 = 1e-30
    decay_rate = -0.8
    p = torch.nn.Parameter(torch.randn(5))
    opt = Adafactor(
        [p],
        lr=1e-3,
        beta1=None,
        beta2=0.99,
        eps=(eps0, 1e-3),
        scale_parameter=False,
        relative_step=False,
        weight_decay=0.0,
        factored=False,
        decay_rate=decay_rate,
        stochastic_accumulation=False,
        stochastic_rounding=False,
    )
    v = torch.zeros(5)
    for step in range(1, 101):
        grad = torch.randn(5)
        p.grad = grad.clone()
        opt.step()
        beta2t = 1.0 - math.pow(step, decay_rate)
        v = beta2t * v + (1.0 - beta2t) * (grad ** 2 + eps0)
        if step == 1:
            assert beta2t == pytest.approx(0.0)
            assert torch.equal(opt.state[p]["exp_avg_sq"], grad ** 2 + eps0)
        if step == 10:
            assert beta2t == pytest.approx(0.842, abs=1e-3)
            # rms_max_decay_rate (0.97) at step 10 is negative and must not match.
            wrong = 1.0 - math.pow(step, 0.97)
            assert wrong < 0
            assert beta2t != pytest.approx(wrong)
        if step in (1, 10, 100):
            assert torch.allclose(opt.state[p]["exp_avg_sq"], v, rtol=1e-5, atol=0.0)


def test_get_optimizer_drops_adaptive_beta2_keys():
    p = torch.nn.Parameter(torch.ones(2))
    opt = get_optimizer(
        [p],
        optimizer_type="adafactor",
        learning_rate=1e-4,
        optimizer_params={
            "beta2": 0.99,
            "beta2_adaptive": True,
            "beta2_min": 0.9,
            "relative_step": False,
            "scale_parameter": False,
            "beta1": None,
        },
    )
    assert isinstance(opt, Adafactor)
    for group in opt.param_groups:
        assert "beta2_adaptive" not in group
        assert "beta2_min" not in group
        assert group["decay_rate"] == pytest.approx(-0.8)
        assert group["beta2"] == pytest.approx(0.99)


def test_resume_pops_stale_group_beta2_effective_and_gns():
    opt, p = _make_opt(0.97)
    p.grad = torch.ones_like(p) * 0.1
    opt.step()
    assert opt.get_mean_beta2() == pytest.approx(0.97)
    sd = opt.state_dict()
    sd["param_groups"][0]["beta2"] = 0.5
    sd["param_groups"][0]["beta2_effective"] = 0.0
    sd["param_groups"][0]["gns"] = 7.0
    opt.load_state_dict(sd)
    assert "beta2_effective" not in opt.param_groups[0]
    assert "gns" not in opt.param_groups[0]
    assert opt.get_mean_beta2() == pytest.approx(0.97)
