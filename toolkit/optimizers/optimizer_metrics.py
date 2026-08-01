"""Optimizer-agnostic step metrics from param / grad / Δp (outside optimizer.step)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch


class OptimizerStepMetrics:
    """
    Collect generic training metrics around ``optimizer.step()``.

    Call ``before_step`` then ``after_step``. Own per-group state (not written into
    ``optimizer.param_groups``). Works with any ``torch.optim.Optimizer`` including
    local Adafactor and HFAdafactor.
    """

    def __init__(self, rms_max_decay_rate: float = 0.97, eps: float = 1e-30):
        rms_max_decay_rate = float(rms_max_decay_rate)
        if not (0.0 < rms_max_decay_rate <= 1.0):
            raise ValueError(
                f"rms_max_decay_rate must satisfy 0 < rms_max_decay_rate <= 1, "
                f"got rms_max_decay_rate={rms_max_decay_rate}"
            )
        self.rms_max_decay_rate = rms_max_decay_rate
        self.eps = float(eps)
        self._groups: List[dict] = []
        self._snapshots: Dict[int, torch.Tensor] = {}
        self._before_rows: Optional[List[list]] = None

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _param_fp32(p: torch.nn.Parameter) -> torch.Tensor:
        data = p.data
        if data.dtype != torch.float32:
            return data.float()
        return data

    @staticmethod
    def _scalar_item(store: dict, key: str, default: float = 0.0) -> float:
        t = store.get(key)
        if t is None:
            return default
        return t.item() if isinstance(t, torch.Tensor) else float(t)

    @staticmethod
    def _running_max_update(store: dict, key: str, candidate: torch.Tensor) -> None:
        candidate = candidate.detach()
        if key not in store:
            store[key] = candidate.clone()
            return
        current = store[key]
        if isinstance(current, torch.Tensor) and current.device != candidate.device:
            current = current.to(candidate.device)
            store[key] = current
        store[key] = torch.maximum(current, candidate)

    @staticmethod
    def _running_min_update(store: dict, key: str, candidate: torch.Tensor) -> None:
        candidate = candidate.detach()
        if key not in store:
            store[key] = candidate.clone()
            return
        current = store[key]
        if isinstance(current, torch.Tensor) and current.device != candidate.device:
            current = current.to(candidate.device)
            store[key] = current
        store[key] = torch.minimum(current, candidate)

    @staticmethod
    def _maybe_running_max(store: dict, key: str, candidate: torch.Tensor) -> None:
        c = candidate.detach().item()
        if not math.isfinite(c):
            return
        OptimizerStepMetrics._running_max_update(store, key, candidate)

    @staticmethod
    def _maybe_running_min(store: dict, key: str, candidate: torch.Tensor) -> None:
        c = candidate.detach().item()
        if not math.isfinite(c):
            return
        OptimizerStepMetrics._running_min_update(store, key, candidate)

    @staticmethod
    def _mean_per_group(per_group: List[float]) -> float:
        if len(per_group) == 0:
            return 0.0
        return torch.tensor(per_group, dtype=torch.float64).mean().item()

    def _ensure_groups(self, optimizer: torch.optim.Optimizer) -> None:
        n = len(optimizer.param_groups)
        if len(self._groups) != n:
            self._groups = [{} for _ in range(n)]
        for gi, group in enumerate(optimizer.param_groups):
            name = group.get("name")
            if name is not None:
                self._groups[gi]["name"] = name

    @torch.no_grad()
    def before_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Snapshot params and record param/grad RMS (call immediately before step)."""
        self._ensure_groups(optimizer)
        self._snapshots = {}
        self._before_rows = [[] for _ in optimizer.param_groups]
        dr = self.rms_max_decay_rate

        for gi, group in enumerate(optimizer.param_groups):
            gs = self._groups[gi]
            if "rms_max" in gs:
                gs["rms_max"] = gs["rms_max"] * dr
            if "rms_min" in gs:
                gs["rms_min"] = gs["rms_min"] / dr
            if "update_rms_max" in gs:
                gs["update_rms_max"] = gs["update_rms_max"] * dr
            if "grad_rms_max" in gs:
                gs["grad_rms_max"] = gs["grad_rms_max"] * dr

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                p_fp32 = self._param_fp32(p)
                snapshot = p_fp32.detach().clone()
                self._snapshots[id(p)] = snapshot

                param_rms = self._rms(snapshot)
                grad = p.grad
                if grad.dtype != torch.float32:
                    grad = grad.to(torch.float32)
                grad_rms = self._rms(grad)

                self._maybe_running_max(gs, "rms_max", param_rms)
                self._maybe_running_min(gs, "rms_min", param_rms)
                self._maybe_running_max(gs, "grad_rms_max", grad_rms)

                self._before_rows[gi].append(
                    (p, param_rms.detach(), grad_rms.detach(), p.numel())
                )

            self._finalize_before_group(gi)

    def _finalize_before_group(self, gi: int) -> None:
        rows = self._before_rows[gi] if self._before_rows is not None else []
        gs = self._groups[gi]
        if not rows:
            return
        ref_device = rows[0][0].device
        pr_values = []
        gr_values = []
        weights = []
        for _p, param_rms, grad_rms, numel in rows:
            pr_values.append(torch.as_tensor(param_rms, device=ref_device, dtype=torch.float32))
            gr_values.append(torch.as_tensor(grad_rms, device=ref_device, dtype=torch.float32))
            weights.append(numel)
        w = torch.tensor(weights, device=ref_device, dtype=torch.float32)
        pr = torch.stack(pr_values)
        gr = torch.stack(gr_values)
        avg_rms = (torch.sum(pr * w) / (torch.sum(w) + 1e-12)).item()
        avg_gr = (torch.sum(gr * w) / (torch.sum(w) + 1e-12)).item()
        gs["param_rms"] = torch.tensor(avg_rms, dtype=torch.float32, device=ref_device)
        gs["grad_rms"] = torch.tensor(avg_gr, dtype=torch.float32, device=ref_device)

        if "rms_ema" not in gs:
            gs["rms_ema"] = torch.tensor(avg_rms, dtype=torch.float32, device=ref_device)
        else:
            prev = self._scalar_item(gs, "rms_ema", 0.0)
            gs["rms_ema"] = torch.tensor(
                prev * self.rms_max_decay_rate + avg_rms * (1.0 - self.rms_max_decay_rate),
                dtype=torch.float32,
                device=ref_device,
            )

    @torch.no_grad()
    def after_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Compute update RMS from Δp and finalize derived metrics (call after step)."""
        self._ensure_groups(optimizer)
        if self._before_rows is None:
            self._snapshots.clear()
            return

        for gi, group in enumerate(optimizer.param_groups):
            rows = self._before_rows[gi]
            gs = self._groups[gi]
            if not rows:
                continue

            ref_device = rows[0][0].device
            ur_values = []
            ur_weights = []
            for p, _param_rms, _grad_rms, numel in rows:
                snap = self._snapshots.pop(id(p), None)
                if snap is None:
                    continue
                p_after = self._param_fp32(p).detach()
                if snap.device != p_after.device:
                    snap = snap.to(p_after.device)
                if snap.dtype != p_after.dtype:
                    snap = snap.to(dtype=p_after.dtype)
                update_rms = self._rms(snap - p_after)
                self._maybe_running_max(gs, "update_rms_max", update_rms)
                ur_values.append(
                    torch.as_tensor(update_rms, device=ref_device, dtype=torch.float32).reshape(())
                )
                ur_weights.append(numel)

            if not ur_values:
                continue

            ur = torch.stack(ur_values)
            w = torch.tensor(ur_weights, device=ref_device, dtype=torch.float32)
            avg_ur = (torch.sum(ur * w) / (torch.sum(w) + 1e-12)).item()
            gs["update_rms"] = torch.tensor(avg_ur, dtype=torch.float32, device=ref_device)

            eps_t = torch.tensor(self.eps, dtype=torch.float32, device=ref_device)
            u_rms_t = gs["update_rms"]
            u_max = gs.get("update_rms_max")
            if u_max is None:
                u_max = u_rms_t
            elif isinstance(u_max, torch.Tensor):
                u_max = u_max.to(ref_device)
            else:
                u_max = torch.tensor(float(u_max), dtype=torch.float32, device=ref_device)
            g_mean_t = gs.get("grad_rms")
            if g_mean_t is None:
                g_mean_t = torch.tensor(0.0, dtype=torch.float32, device=ref_device)
            elif isinstance(g_mean_t, torch.Tensor):
                g_mean_t = g_mean_t.to(ref_device)
            else:
                g_mean_t = torch.tensor(float(g_mean_t), dtype=torch.float32, device=ref_device)
            gs["step_efficiency"] = u_rms_t / (u_max + eps_t)
            gs["dynamic_gain"] = u_rms_t / (g_mean_t + eps_t)

        self._snapshots.clear()
        self._before_rows = None

    # --- getters (same surface as former Adafactor generic metrics) ---

    def get_update_rms(self) -> List[float]:
        return [self._scalar_item(gs, "update_rms", 0.0) for gs in self._groups]

    def get_update_rms_max(self) -> List[float]:
        return [self._scalar_item(gs, "update_rms_max", 0.0) for gs in self._groups]

    def get_rms(self) -> List[float]:
        return [self._scalar_item(gs, "param_rms", 0.0) for gs in self._groups]

    def get_mean_rms(self) -> float:
        return self._mean_per_group(self.get_rms())

    def get_max_rms(self) -> dict:
        out = {}
        for gs in self._groups:
            name = gs.get("name")
            if name is None or "rms_max" not in gs:
                continue
            out[name] = self._scalar_item(gs, "rms_max", 0.0)
        return out

    def get_min_rms(self) -> float:
        per_group = [
            self._scalar_item(gs, "rms_min", 0.0)
            for gs in self._groups
            if "rms_min" in gs
        ]
        if len(per_group) == 0:
            return 0.0
        return torch.tensor(per_group, dtype=torch.float64).min().item()

    def get_mean_update_rms(self) -> float:
        return self._mean_per_group(self.get_update_rms())

    def get_mean_update_rms_max(self) -> float:
        return self._mean_per_group(self.get_update_rms_max())

    def get_dynamic_gain(self) -> List[float]:
        return [self._scalar_item(gs, "dynamic_gain", 0.0) for gs in self._groups]

    def get_mean_dynamic_gain(self) -> float:
        return self._mean_per_group(self.get_dynamic_gain())

    def get_grad_rms(self) -> List[float]:
        return [self._scalar_item(gs, "grad_rms", 0.0) for gs in self._groups]

    def get_grad_rms_max(self) -> List[float]:
        return [self._scalar_item(gs, "grad_rms_max", 0.0) for gs in self._groups]

    def get_mean_grad_rms(self) -> float:
        return self._mean_per_group(self.get_grad_rms())

    def get_mean_grad_rms_max(self) -> float:
        return self._mean_per_group(self.get_grad_rms_max())

    def get_step_efficiency(self) -> List[float]:
        return [self._scalar_item(gs, "step_efficiency", 0.0) for gs in self._groups]

    def get_mean_step_efficiency(self) -> float:
        return self._mean_per_group(self.get_step_efficiency())
