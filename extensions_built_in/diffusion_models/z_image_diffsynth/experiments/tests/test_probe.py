"""probe helpers (no GPU)."""

from __future__ import annotations

from ._loaders import load_probe


def test_cuda_gate_not_venv():
    probe = load_probe()
    assert probe.cuda_gate_reason(device="cuda", cuda_available=True, is_venv=False) == "not_venv"


def test_cuda_gate_ok():
    probe = load_probe()
    assert probe.cuda_gate_reason(device="cuda", cuda_available=True, is_venv=True) is None


def test_child_wall_timeout():
    probe = load_probe()
    load, prefix, n_forks, measure, step_t, sample = 180.0, 100, 10, 10, 30.0, 0.0
    n_new = prefix + n_forks * measure
    assert probe.child_wall_timeout_s(load, n_new, step_t, sample) == load + n_new * step_t + sample


def test_step_timeout_skips_first_and_last():
    probe = load_probe()
    assert probe.step_timeout_exceeded(
        100.0, is_first=True, is_sample_or_save=False, is_last=False, limit_s=1.0
    ) is False
    assert probe.step_timeout_exceeded(
        100.0, is_first=False, is_sample_or_save=False, is_last=True, limit_s=1.0
    ) is False
    assert probe.step_timeout_exceeded(
        100.0, is_first=False, is_sample_or_save=False, is_last=False, limit_s=1.0
    ) is True
