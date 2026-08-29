"""Dynamic train-loop bound: inspect source + tiny simulator (CPU, no run())."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


def _run_source() -> str:
    return inspect.getsource(BaseSDTrainProcess.run)


def test_run_source_uses_dynamic_while_bound():
    src = _run_source()
    assert "while self.step_num < self.train_config.steps" in src
    assert "start_step_num" in src
    assert "keep_training" not in src
    assert "while True" not in src
    assert "self.save()" in src
    assert "end_training" in src
    assert "del (" in src


def _simulate(proc, *, max_iters: int = 50) -> int:
    n_iters = 0
    start_step_num = proc.step_num
    while proc.step_num < proc.train_config.steps:
        step = proc.step_num
        proc.step_num = step
        n_iters += 1
        proc.step_num = step + 1
        proc.end_step_hook()
        if n_iters >= max_iters:
            break
    return n_iters


def test_hook_raise_steps_extends_loop():
    proc = SimpleNamespace(
        step_num=0,
        train_config=SimpleNamespace(steps=2),
    )

    def end_step_hook():
        if proc.step_num == 2:
            proc.train_config.steps = 4

    proc.end_step_hook = end_step_hook
    n_iters = _simulate(proc)
    assert n_iters == 4
    assert proc.step_num == 4
    assert proc.train_config.steps == 4


def test_hook_rewind_once_then_exit():
    proc = SimpleNamespace(
        step_num=0,
        train_config=SimpleNamespace(steps=4),
    )
    rewinds = {"n": 0}

    def end_step_hook():
        if proc.step_num == 4 and rewinds["n"] == 0:
            proc.step_num = 2
            rewinds["n"] += 1

    proc.end_step_hook = end_step_hook
    n_iters = _simulate(proc)
    assert rewinds["n"] == 1
    assert n_iters == 6
    assert n_iters < 50
    assert proc.step_num == 4
    assert proc.train_config.steps == 4
