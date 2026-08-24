"""probe child streaming and step_s parse."""

from __future__ import annotations

import os
import sys
import time

import pytest

from ._loaders import load_probe


@pytest.fixture(scope="module")
def probe():
    return load_probe()


def test_median_step_s_empty(probe):
    assert probe.median_step_s("") is None
    assert probe.median_step_s("no timings here\n") is None


def test_median_step_s_median(probe):
    log = (
        "tune: step=2 step_s=3.000\n"
        "tune: step=3 step_s=1.000\n"
        "tune: step=4 step_s=2.000\n"
        "other line\n"
    )
    assert probe.median_step_s(log) == pytest.approx(2.0)


def test_run_child_streams_and_captures(probe, capsys):
    code, out = probe._run_child(
        [sys.executable, "-u", "-c", "print('hello_tune_step')"],
        dict(os.environ),
        os.getcwd(),
    )
    assert code == 0
    assert "hello_tune_step" in out
    captured = capsys.readouterr()
    assert "hello_tune_step" in captured.out


def test_step_timeout_exceeded_helper(probe):
    assert probe.step_timeout_exceeded(
        1.001, is_first=False, is_sample_or_save=False, is_last=False, limit_s=1.0
    )
    assert not probe.step_timeout_exceeded(
        1.0, is_first=False, is_sample_or_save=False, is_last=False, limit_s=1.0
    )
    assert not probe.step_timeout_exceeded(
        2.0, is_first=True, is_sample_or_save=False, is_last=False, limit_s=1.0
    )
    assert not probe.step_timeout_exceeded(
        2.0, is_first=False, is_sample_or_save=True, is_last=False, limit_s=1.0
    )
    assert not probe.step_timeout_exceeded(
        2.0, is_first=False, is_sample_or_save=False, is_last=True, limit_s=1.0
    )


def test_child_wall_timeout_s(probe):
    assert probe.child_wall_timeout_s(180, 20, 1.0, 60) == pytest.approx(260.0)


def test_cuda_gate_reason(probe):
    assert (
        probe.cuda_gate_reason(device="cuda", cuda_available=False, is_venv=True)
        == "cuda_unavailable"
    )
    assert (
        probe.cuda_gate_reason(device="cpu", cuda_available=True, is_venv=True)
        == "not_cuda_device"
    )
    assert (
        probe.cuda_gate_reason(device="cuda", cuda_available=True, is_venv=False)
        == "not_venv"
    )
    assert (
        probe.cuda_gate_reason(device="cuda", cuda_available=True, is_venv=True)
        is None
    )


def test_cuda_placement_reason(probe):
    assert (
        probe.cuda_placement_reason(
            param_device_type="cpu", payload_device_type=None, mem_gb=8.0
        )
        == "not_cuda_device"
    )
    assert (
        probe.cuda_placement_reason(
            param_device_type="cuda", payload_device_type="cpu", mem_gb=8.0
        )
        == "not_cuda_device"
    )
    assert (
        probe.cuda_placement_reason(
            param_device_type="cuda", payload_device_type="cuda", mem_gb=0.1
        )
        == "not_cuda_device"
    )
    assert (
        probe.cuda_placement_reason(
            param_device_type="cuda", payload_device_type="cuda", mem_gb=4.0
        )
        is None
    )
    assert (
        probe.cuda_placement_reason(
            param_device_type="cuda", payload_device_type=None, mem_gb=4.0
        )
        is None
    )


def test_run_child_timeout_kills(probe):
    t0 = time.perf_counter()
    code, out = probe._run_child(
        [sys.executable, "-u", "-c", "import time; time.sleep(5)"],
        dict(os.environ),
        os.getcwd(),
        0.3,
    )
    elapsed = time.perf_counter() - t0
    assert code != 0
    assert elapsed < 4.0
    assert "child_wall_timeout" in out
