"""session snapshot/restore, knobs, reseed, end_step_hook (CPU)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ._loaders import load_session


class FakeOpt:
    def __init__(self, lr=1e-4, beta2=0.99):
        self._lr = float(lr)
        self._beta2 = float(beta2)
        self.buf = torch.ones(3, 3)

    def state_dict(self):
        return {"buf": self.buf, "lr": self._lr, "beta2": self._beta2}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or "buf" not in state:
            raise ValueError("incompatible state_dict")
        self.buf = state["buf"].detach().clone()
        if "lr" in state:
            self._lr = float(state["lr"])
        if "beta2" in state:
            self._beta2 = float(state["beta2"])

    def set_lr(self, value):
        self._lr = float(value)

    def set_beta2(self, value):
        self._beta2 = float(value)


class NestedOpt:
    def __init__(self, inner):
        self.optimizer = inner


class ProgressBar:
    def __init__(self, total, n=0):
        self.total = total
        self.n = n


def _trainer(tmp: Path, *, prefix, measure, forks, steps=None):
    warm = tmp / "warm"
    proc = SimpleNamespace()
    proc.config = {
        "experiments": {
            "prefix_steps": prefix,
            "measure_steps": measure,
            "warm_training_folder": str(warm),
            "forks": forks,
        }
    }
    proc.job = SimpleNamespace(name="probe")
    proc.training_folder = str(tmp / "prefix_run")
    proc.save_root = os.path.join(proc.training_folder, proc.job.name)
    proc.train_config = SimpleNamespace(steps=prefix if steps is None else steps)
    proc.start_step = 0
    proc.step_num = 0
    proc.network = nn.Linear(4, 3)
    inner = FakeOpt()
    proc.optimizer = NestedOpt(inner)
    proc.progress_bar = ProgressBar(proc.train_config.steps)
    proc.training_seed = 4
    proc.save_calls = []
    proc.load_model_calls = 0

    def save(step=None):
        os.makedirs(proc.save_root, exist_ok=True)
        (Path(proc.save_root) / f"{proc.job.name}.safetensors").write_bytes(b"x")
        proc.save_calls.append(proc.save_root)

    def load_model():
        proc.load_model_calls += 1

    proc.save = save
    proc.load_model = load_model
    return proc


def _install(monkeypatch, sess, orig=None):
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

    if orig is None:
        def orig(self, *args, **kwargs):
            pass

    monkeypatch.setattr(BaseSDTrainProcess, "end_step_hook", orig)
    sess.install_session_hook()
    return BaseSDTrainProcess.end_step_hook


def test_lora_roundtrip_same_param_id():
    sess = load_session()
    lin = nn.Linear(4, 3)
    pid_w, pid_b = id(lin.weight), id(lin.bias)
    snap = sess.snapshot_lora(lin)
    sess.restore_lora(lin, snap)
    assert id(lin.weight) == pid_w
    assert id(lin.bias) == pid_b


def test_lora_restore_matches_snapshot_not_mutated():
    sess = load_session()
    lin = nn.Linear(4, 3)
    pid = id(lin.weight)
    before = lin.weight.detach().clone()
    snap = sess.snapshot_lora(lin)
    lin.weight.data.add_(5.0)
    assert not torch.allclose(lin.weight, before)
    sess.restore_lora(lin, snap)
    assert torch.allclose(lin.weight, before)
    assert id(lin.weight) == pid


def test_optimizer_restore_and_knobs():
    sess = load_session()
    inner = FakeOpt(lr=1e-4, beta2=0.99)
    opt = NestedOpt(inner)
    snap = sess.snapshot_optimizer(opt)
    snap_buf = snap["buf"].clone()
    inner.buf.add_(3.0)
    inner.set_lr(9.0)
    inner.set_beta2(0.5)
    sess.restore_optimizer(opt, snap)
    assert torch.allclose(inner.buf, snap_buf)
    sess.set_lr(opt, 4e-4)
    sess.set_beta2(opt, 0.9)
    assert inner._lr == pytest.approx(4e-4)
    assert inner._beta2 == pytest.approx(0.9)


def test_reseed_repeatable():
    sess = load_session()
    sess.reseed(123)
    a = torch.rand(1)
    sess.reseed(123)
    b = torch.rand(1)
    assert torch.equal(a, b)


def test_broken_state_dict_raises():
    sess = load_session()

    class BrokenSnap:
        def state_dict(self):
            raise RuntimeError("broken state_dict")

        def load_state_dict(self, state):
            pass

    with pytest.raises(Exception):
        sess.snapshot_optimizer(BrokenSnap())

    opt = FakeOpt()
    with pytest.raises(Exception):
        sess.restore_optimizer(opt, {"not": "compatible"})


def test_hook_prefix_measure_and_empty(tmp_path, monkeypatch):
    sess = load_session()
    hook = _install(monkeypatch, sess)
    prefix, measure = 10, 5
    fork0 = {
        "id": "continue",
        "lr": 1e-4,
        "beta2": 0.99,
        "training_folder": str(tmp_path / "fork_continue"),
    }
    fork1 = {
        "id": "lr_x4",
        "lr": 4e-4,
        "beta2": 0.9,
        "training_folder": str(tmp_path / "fork_lr_x4"),
    }
    proc = _trainer(tmp_path, prefix=prefix, measure=measure, forks=[fork0, fork1])
    weight_id = id(proc.network.weight)

    proc.step_num = prefix
    hook(proc)
    warm_root = os.path.join(str(tmp_path / "warm"), "probe")
    assert proc.save_calls[0] == warm_root
    assert (Path(warm_root) / "probe.safetensors").is_file()
    assert proc.train_config.steps == prefix + measure
    assert proc.start_step == prefix
    assert proc.progress_bar.total == prefix + measure
    inner = proc.optimizer.optimizer
    assert inner._lr == pytest.approx(1e-4)
    assert inner._beta2 == pytest.approx(0.99)
    assert proc.save_root == os.path.join(fork0["training_folder"], "probe")
    assert id(proc.network.weight) == weight_id
    assert proc.load_model_calls == 0

    proc.step_num = prefix + measure
    bound = proc.train_config.steps
    hook(proc)
    assert proc.save_calls[1] == os.path.join(fork0["training_folder"], "probe")
    assert (Path(fork0["training_folder"]) / "probe" / "probe.safetensors").is_file()
    assert proc.step_num == prefix
    assert proc.start_step == prefix
    assert proc.progress_bar.n == prefix
    assert inner._lr == pytest.approx(4e-4)
    assert inner._beta2 == pytest.approx(0.9)
    assert proc.save_root == os.path.join(fork1["training_folder"], "probe")
    assert proc.train_config.steps == bound
    assert proc.load_model_calls == 0

    proc.step_num = prefix + measure
    hook(proc)
    assert proc.save_calls[2] == os.path.join(fork1["training_folder"], "probe")
    assert proc.step_num == prefix + measure
    assert proc.train_config.steps == bound
    assert proc.load_model_calls == 0

    empty = _trainer(
        tmp_path / "empty",
        prefix=prefix,
        measure=measure,
        forks=[],
        steps=prefix,
    )
    empty.step_num = prefix
    empty_steps = empty.train_config.steps
    hook(empty)
    assert len(empty.save_calls) == 1
    assert empty.save_calls[0] == os.path.join(str(tmp_path / "empty" / "warm"), "probe")
    assert empty.train_config.steps == empty_steps
    assert empty.load_model_calls == 0


def test_stamp_reset_after_slow_save(tmp_path, monkeypatch):
    sess = load_session()
    tripped = []

    def fake_timer(self, *args, **kwargs):
        now = time.perf_counter()
        last = getattr(self, "_zimage_exp_step_last", None)
        if last is not None and (now - last) > 0.1:
            tripped.append(now - last)
        self._zimage_exp_step_last = now

    hook = _install(monkeypatch, sess, orig=fake_timer)
    prefix = 4
    proc = _trainer(tmp_path, prefix=prefix, measure=2, forks=[])
    real_save = proc.save

    def slow_save(step=None):
        time.sleep(0.2)
        real_save(step)

    proc.save = slow_save
    proc.step_num = prefix
    hook(proc)
    proc.step_num = prefix + 1
    hook(proc)
    assert tripped == []
