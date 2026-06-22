"""Effective LR (|update|/|grad|) behavior when grad shrinks — not explicit _get_lr."""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testing"))

from simulate_user_regime_effective_lr import USER, make_state, step_and_decompose


def _warmup_and_decay(param_rms=0.01, decay_steps=50, decay=0.92):
    shape = (16, 32)
    d = torch.randn(shape)
    d = d / d.norm()
    group = dict(USER)
    state = make_state(shape, param_rms)
    for _ in range(200):
        step_and_decompose(d * 1e-4, state, group)
    g = 1e-7
    first = last = None
    for _ in range(decay_steps):
        m = step_and_decompose(d * g, state, group)
        if first is None:
            first = m
        last = m
        g *= decay
    return first, last


def test_explicit_lr_flat_when_grad_decays():
    first, last = _warmup_and_decay()
    assert math.isclose(first["lr"], last["lr"], rel_tol=1e-3)


def test_eff_total_drops_during_sustained_grad_decay_with_momentum():
    first, last = _warmup_and_decay()
    assert last["eff_total"] < first["eff_total"]


def test_eff_total_stable_without_momentum_on_same_decay():
    shape = (16, 32)
    d = torch.randn(shape)
    d = d / d.norm()
    group = dict(USER)
    group["beta1"] = None
    state = make_state(shape, 0.01)
    for _ in range(200):
        step_and_decompose(d * 1e-4, state, group)
    g = 1e-7
    effs = []
    for _ in range(30):
        m = step_and_decompose(d * g, state, group)
        effs.append(m["eff_total"])
        g *= 0.92
    # no momentum: eff tracks preconditioner * lr, no memory-driven drop
    assert effs[-1] >= effs[0] * 0.5


def test_momentum_amplifies_final_vs_scaled():
    shape = (16, 32)
    d = torch.randn(shape)
    d = d / d.norm()
    group = dict(USER)
    state = make_state(shape, 0.01)
    for _ in range(200):
        step_and_decompose(d * 1e-4, state, group)
    m = step_and_decompose(d * 1e-8, state, group)
    assert m["eff_momentum"] > 10.0
