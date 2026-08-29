"""RAM snapshot/restore of LoRA + optimizer; session end_step_hook."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from toolkit.accelerator import unwrap_model


def _clone_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, Mapping):
        return {k: _clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_tree(v) for v in obj)
    return obj


def snapshot_lora(network: torch.nn.Module) -> dict[str, Any]:
    return {k: _clone_tree(v) for k, v in network.state_dict().items()}


def restore_lora(network: torch.nn.Module, state: Mapping[str, Any]) -> None:
    network.load_state_dict(state)


def unwrap_optimizer(optimizer: Any) -> Any:
    optimizer = unwrap_model(optimizer)
    while getattr(optimizer, "optimizer", None) is not None:
        optimizer = optimizer.optimizer
    return optimizer


def snapshot_optimizer(optimizer: Any) -> Any:
    opt = unwrap_optimizer(optimizer)
    return _clone_tree(opt.state_dict())


def restore_optimizer(optimizer: Any, state: Any) -> None:
    opt = unwrap_optimizer(optimizer)
    opt.load_state_dict(state)


def set_lr(optimizer: Any, lr: float) -> None:
    unwrap_optimizer(optimizer).set_lr(float(lr))


def set_beta2(optimizer: Any, beta2: float) -> None:
    unwrap_optimizer(optimizer).set_beta2(float(beta2))


def reseed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _payload(proc) -> Mapping[str, Any] | None:
    cfg = getattr(proc, "config", None)
    if not isinstance(cfg, Mapping):
        return None
    exp = cfg.get("experiments")
    if not isinstance(exp, Mapping):
        return None
    if "prefix_steps" not in exp or "measure_steps" not in exp:
        return None
    return exp


def _seed_for(proc, payload: Mapping[str, Any], fork: Mapping[str, Any] | None) -> int:
    seed = getattr(proc, "training_seed", None)
    if seed is not None:
        return int(seed)
    if fork is not None and fork.get("seed") is not None:
        return int(fork["seed"])
    if payload.get("training_seed") is not None:
        return int(payload["training_seed"])
    return 0


def _set_save_root(proc, training_folder: str) -> None:
    proc.training_folder = training_folder
    proc.save_root = os.path.join(training_folder, proc.job.name)


def _enter_fork(proc, payload: Mapping[str, Any], fork: Mapping[str, Any]) -> None:
    restore_lora(proc.network, proc._zimage_exp_lora_snap)
    restore_optimizer(proc.optimizer, proc._zimage_exp_opt_snap)
    reseed(_seed_for(proc, payload, fork))
    set_lr(proc.optimizer, fork["lr"])
    set_beta2(proc.optimizer, fork["beta2"])
    _set_save_root(proc, fork["training_folder"])


def _handle_prefix(proc, payload: Mapping[str, Any], prefix: int, measure: int, forks: list) -> None:
    _set_save_root(proc, payload["warm_training_folder"])
    proc.save()
    proc._zimage_exp_lora_snap = snapshot_lora(proc.network)
    proc._zimage_exp_opt_snap = snapshot_optimizer(proc.optimizer)
    proc.start_step = prefix
    if not forks:
        return
    proc._zimage_exp_fork_i = 0
    _enter_fork(proc, payload, forks[0])
    proc.train_config.steps = prefix + measure
    pb = getattr(proc, "progress_bar", None)
    if pb is not None:
        pb.total = proc.train_config.steps


def _handle_measure(proc, payload: Mapping[str, Any], prefix: int, forks: list) -> None:
    proc.save()
    nxt = int(getattr(proc, "_zimage_exp_fork_i", 0)) + 1
    if nxt >= len(forks):
        return
    proc._zimage_exp_fork_i = nxt
    _enter_fork(proc, payload, forks[nxt])
    proc.step_num = prefix
    proc.start_step = prefix
    pb = getattr(proc, "progress_bar", None)
    if pb is not None:
        pb.n = prefix


def install_session_hook() -> None:
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

    orig = BaseSDTrainProcess.end_step_hook
    if getattr(orig, "_zimage_exp_session", False):
        return

    def hooked(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        payload = _payload(self)
        if payload is None:
            return
        prefix = int(payload["prefix_steps"])
        measure = int(payload["measure_steps"])
        forks = list(payload.get("forks") or [])
        step_num = int(self.step_num)
        boundary = False
        if step_num == prefix and not getattr(self, "_zimage_exp_prefix_done", False):
            self._zimage_exp_prefix_done = True
            boundary = True
            _handle_prefix(self, payload, prefix, measure, forks)
        elif forks and step_num == prefix + measure:
            boundary = True
            _handle_measure(self, payload, prefix, forks)
        if boundary:
            self._zimage_exp_step_last = time.perf_counter()

    hooked._zimage_exp_session = True  # type: ignore[attr-defined]
    BaseSDTrainProcess.end_step_hook = hooked  # type: ignore[method-assign]
