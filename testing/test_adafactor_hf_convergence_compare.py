"""
Experimental comparison: local toolkit Adafactor vs HuggingFace transformers Adafactor.

Focuses on algorithmic core (not metrics / stochastic extras).
Local stochastic_rounding/accumulation are disabled for a fair step comparison.

Scenarios
---------
A) Manual LR (HF T5-style): relative_step=False, scale_parameter=False.
   Isolates the main remaining core difference: fixed beta2 (local) vs
   time-decay beta2t = 1 - step**decay_rate (HF).

B) HF paper-style relative schedule vs local fixed LR:
   HF: relative_step=True, scale_parameter=True, lr=None
   Local: relative_step=False, scale_parameter=False, lr=1e-3
   (toolkit default product path vs HF defaults)

C) Local fork relative_step=True (RMS-ratio adaptive, NOT HF 1/sqrt(t))
   vs HF relative_step=True on the same problem.

Run:
  venv\\Scripts\\python.exe -m pytest testing/test_adafactor_hf_convergence_compare.py -s -v
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from transformers.optimization import Adafactor as HFAdafactor

from toolkit.optimizers.adafactor import Adafactor as LocalAdafactor


torch.set_num_threads(1)


@dataclass
class RunResult:
    name: str
    final_loss: float
    best_loss: float
    steps_to_1e_3: Optional[int]
    steps_to_1e_4: Optional[int]
    wall_s: float
    loss_curve: List[float]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def _make_local(
    params,
    *,
    lr: float,
    relative_step: bool = False,
    scale_parameter: bool = False,
    beta2: float = 0.99,
    beta1=None,
    weight_decay: float = 0.0,
    clip_threshold: float = 1.0,
) -> LocalAdafactor:
    return LocalAdafactor(
        params,
        lr=lr,
        eps=(1e-30, 1e-3),
        clip_threshold=clip_threshold,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        scale_parameter=scale_parameter,
        relative_step=relative_step,
        warmup_init=False,
        stochastic_accumulation=False,
        stochastic_rounding=False,
        factored=None,
        beta2_adaptive=False,
    )


def _make_hf(
    params,
    *,
    lr: Optional[float],
    relative_step: bool = False,
    scale_parameter: bool = False,
    decay_rate: float = -0.8,
    beta1=None,
    weight_decay: float = 0.0,
    clip_threshold: float = 1.0,
) -> HFAdafactor:
    return HFAdafactor(
        params,
        lr=lr,
        eps=(1e-30, 1e-3),
        clip_threshold=clip_threshold,
        decay_rate=decay_rate,
        beta1=beta1,
        weight_decay=weight_decay,
        scale_parameter=scale_parameter,
        relative_step=relative_step,
        warmup_init=False,
    )


def _clone_module(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)


def _train(
    name: str,
    module: nn.Module,
    make_opt: Callable[[List[nn.Parameter]], torch.optim.Optimizer],
    loss_fn: Callable[[nn.Module], torch.Tensor],
    steps: int,
) -> RunResult:
    opt = make_opt(list(module.parameters()))
    losses: List[float] = []
    best = math.inf
    t0 = time.perf_counter()
    hit_1e3: Optional[int] = None
    hit_1e4: Optional[int] = None

    for t in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(module)
        loss.backward()
        opt.step()
        v = float(loss.detach())
        if not math.isfinite(v):
            v = math.inf
        losses.append(v)
        best = min(best, v)
        if hit_1e3 is None and v <= 1e-3:
            hit_1e3 = t
        if hit_1e4 is None and v <= 1e-4:
            hit_1e4 = t

    return RunResult(
        name=name,
        final_loss=losses[-1],
        best_loss=best,
        steps_to_1e_3=hit_1e3,
        steps_to_1e_4=hit_1e4,
        wall_s=time.perf_counter() - t0,
        loss_curve=losses,
    )


# ---------------------------------------------------------------------------
# Problem factories
# ---------------------------------------------------------------------------

class _Quadratic(nn.Module):
    """Ill-conditioned diagonal quadratic: f(x)=0.5 (x-x*)^T diag(c) (x-x*)."""

    def __init__(self, n: int = 64, cond: float = 1e4, target_scale: float = 0.0):
        super().__init__()
        # Start away from target so the problem is non-trivial.
        self.x = nn.Parameter(torch.ones(n))
        c = torch.logspace(0, math.log10(cond), n)
        self.register_buffer("c", c)
        # target_scale=0 → optimum at 0 (scale_parameter collapses LR as ||θ||→0)
        self.register_buffer("x_star", target_scale * torch.linspace(-1.0, 1.0, n))

    def forward(self) -> torch.Tensor:
        d = self.x - self.x_star
        return 0.5 * (self.c * d.pow(2)).sum()


class _RosenbrockMLP(nn.Module):
    """Small MLP fit to a noisy Rosenbrock-like target on a fixed batch."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        x = torch.linspace(-1.5, 1.5, 64)
        X = torch.stack(torch.meshgrid(x, x, indexing="ij"), dim=-1).reshape(-1, 2)
        # subsample for speed
        idx = torch.randperm(X.shape[0])[:256]
        X = X[idx]
        a, b = X[:, 0], X[:, 1]
        y = ((1 - a) ** 2 + 100 * (b - a**2) ** 2).unsqueeze(1)
        y = y / (y.std() + 1e-8)
        self.register_buffer("X", X)
        self.register_buffer("y", y)

    def forward(self) -> torch.Tensor:
        pred = self.net(self.X)
        return (pred - self.y).pow(2).mean()


class _MatrixFactor(nn.Module):
    """Low-rank factorization: ||UV^T - M||^2 with factored Adafactor state."""

    def __init__(self, m: int = 40, n: int = 30, rank: int = 8):
        super().__init__()
        self.U = nn.Parameter(0.1 * torch.randn(m, rank))
        self.V = nn.Parameter(0.1 * torch.randn(n, rank))
        true_U = torch.randn(m, rank)
        true_V = torch.randn(n, rank)
        self.register_buffer("M", true_U @ true_V.T)

    def forward(self) -> torch.Tensor:
        return (self.U @ self.V.T - self.M).pow(2).mean()


def _fmt_steps(v: Optional[int]) -> str:
    return "-" if v is None else str(v)


def _print_table(title: str, rows: List[RunResult]) -> None:
    print(f"\n=== {title} ===")
    hdr = (
        f"{'name':<28} {'final':>10} {'best':>10} "
        f"{'to1e-3':>8} {'to1e-4':>8} {'wall_s':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.name:<28} {r.final_loss:10.3e} {r.best_loss:10.3e} "
            f"{_fmt_steps(r.steps_to_1e_3):>8} {_fmt_steps(r.steps_to_1e_4):>8} "
            f"{r.wall_s:8.3f}"
        )


def _ratio(a: float, b: float) -> float:
    if b == 0:
        return math.inf if a != 0 else 1.0
    return a / b


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def run_scenario_a_manual_lr(
    seed: int = 0,
    steps: int = 400,
) -> Dict[str, List[RunResult]]:
    """Manual LR: isolates fixed-beta2 (local) vs time-decay beta2 (HF)."""
    problems: List[Tuple[str, Callable[[], nn.Module], int]] = [
        ("quadratic_at_zero", lambda: _Quadratic(64, 1e4, target_scale=0.0), steps),
        ("quadratic_shifted", lambda: _Quadratic(64, 1e4, target_scale=1.0), steps),
        ("rosenbrock_mlp", lambda: _RosenbrockMLP(), steps),
        ("matrix_factor", lambda: _MatrixFactor(), steps),
    ]
    lr = 1e-2
    out: Dict[str, List[RunResult]] = {}

    for pname, factory, n_steps in problems:
        _set_seed(seed)
        base = factory()
        results: List[RunResult] = []

        for label, make in (
            (
                "local_manual",
                lambda ps: _make_local(
                    ps, lr=lr, relative_step=False, scale_parameter=False, beta2=0.99
                ),
            ),
            (
                "hf_manual",
                lambda ps: _make_hf(
                    ps, lr=lr, relative_step=False, scale_parameter=False, decay_rate=-0.8
                ),
            ),
            # Ablation: local with HF-like early beta2 via lower fixed beta2
            (
                "local_beta2_0.9",
                lambda ps: _make_local(
                    ps, lr=lr, relative_step=False, scale_parameter=False, beta2=0.9
                ),
            ),
        ):
            m = _clone_module(base)
            results.append(_train(label, m, make, lambda mod: mod(), n_steps))

        out[pname] = results
        _print_table(f"A manual LR | {pname}", results)

    return out


def run_scenario_b_defaults(
    seed: int = 0,
    steps: int = 500,
) -> Dict[str, List[RunResult]]:
    """HF default relative schedule vs local product-style fixed LR."""
    problems: List[Tuple[str, Callable[[], nn.Module]]] = [
        # shifted: fair for scale_parameter (optimum not at 0)
        ("quadratic_shifted", lambda: _Quadratic(64, 1e4, target_scale=1.0)),
        ("quadratic_at_zero", lambda: _Quadratic(64, 1e4, target_scale=0.0)),
        ("matrix_factor", lambda: _MatrixFactor()),
    ]
    out: Dict[str, List[RunResult]] = {}

    for pname, factory in problems:
        _set_seed(seed)
        base = factory()
        results: List[RunResult] = []

        configs = (
            (
                "hf_relative_default",
                lambda ps: _make_hf(
                    ps, lr=None, relative_step=True, scale_parameter=True
                ),
            ),
            (
                "local_fixed_1e-3",
                lambda ps: _make_local(
                    ps, lr=1e-3, relative_step=False, scale_parameter=False
                ),
            ),
            (
                "local_fixed_1e-2",
                lambda ps: _make_local(
                    ps, lr=1e-2, relative_step=False, scale_parameter=False
                ),
            ),
            (
                "local_scale_only",
                lambda ps: _make_local(
                    ps, lr=1e-2, relative_step=False, scale_parameter=True
                ),
            ),
        )
        for label, make in configs:
            m = _clone_module(base)
            results.append(_train(label, m, make, lambda mod: mod(), steps))

        out[pname] = results
        _print_table(f"B defaults | {pname}", results)

    return out


def run_scenario_c_relative_semantics(
    seed: int = 0,
    steps: int = 500,
) -> Dict[str, List[RunResult]]:
    """Local relative_step (fork) vs HF relative_step (1/sqrt(t))."""
    problems: List[Tuple[str, Callable[[], nn.Module]]] = [
        ("quadratic_shifted", lambda: _Quadratic(64, 1e4, target_scale=1.0)),
        ("matrix_factor", lambda: _MatrixFactor()),
    ]
    out: Dict[str, List[RunResult]] = {}

    for pname, factory in problems:
        _set_seed(seed)
        base = factory()
        results: List[RunResult] = []

        configs = (
            (
                "hf_relative",
                lambda ps: _make_hf(
                    ps, lr=None, relative_step=True, scale_parameter=True
                ),
            ),
            (
                "local_relative_fork",
                lambda ps: _make_local(
                    ps, lr=1e-2, relative_step=True, scale_parameter=True
                ),
            ),
            (
                "local_manual_1e-2",
                lambda ps: _make_local(
                    ps, lr=1e-2, relative_step=False, scale_parameter=False
                ),
            ),
        )
        for label, make in configs:
            m = _clone_module(base)
            results.append(_train(label, m, make, lambda mod: mod(), steps))

        out[pname] = results
        _print_table(f"C relative semantics | {pname}", results)

    return out


def run_early_update_divergence(seed: int = 0) -> Dict[str, float]:
    """
    Quantify early-step algorithmic divergence under identical grads.

    Same init, same grad stream for a few steps; compare ||Δθ|| and param drift.
    Manual LR mode so only beta2 schedule (and tiny numeric extras) differ.
    """
    _set_seed(seed)
    p0 = torch.randn(32, 24)
    g_seq = [torch.randn_like(p0) * 0.1 for _ in range(20)]

    pl = nn.Parameter(p0.clone())
    ph = nn.Parameter(p0.clone())
    ol = _make_local([pl], lr=1e-2, relative_step=False, scale_parameter=False)
    oh = _make_hf([ph], lr=1e-2, relative_step=False, scale_parameter=False)

    drift = []
    upd_l = []
    upd_h = []
    for g in g_seq:
        prev_l = pl.detach().clone()
        prev_h = ph.detach().clone()
        pl.grad = g.clone()
        ph.grad = g.clone()
        ol.step()
        oh.step()
        dl = (pl.detach() - prev_l).norm().item()
        dh = (ph.detach() - prev_h).norm().item()
        upd_l.append(dl)
        upd_h.append(dh)
        drift.append((pl.detach() - ph.detach()).norm().item())

    out = {
        "upd_norm_step1_local": upd_l[0],
        "upd_norm_step1_hf": upd_h[0],
        "upd_norm_step1_ratio_local_over_hf": _ratio(upd_l[0], upd_h[0]),
        "upd_norm_step10_local": upd_l[9],
        "upd_norm_step10_hf": upd_h[9],
        "param_drift_step20": drift[-1],
        "mean_upd_ratio_local_over_hf": sum(_ratio(a, b) for a, b in zip(upd_l, upd_h))
        / len(upd_l),
    }
    print("\n=== Early update divergence (manual LR, identical grads) ===")
    for k, v in out.items():
        print(f"{k}: {v:.6g}")
    return out


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_scenario_a_both_converge_manual_lr(seed: int):
    """Both optimizers must converge on quadratic_at_zero under manual LR."""
    results = run_scenario_a_manual_lr(seed=seed, steps=300)["quadratic_at_zero"]
    by = {r.name: r for r in results}
    assert math.isfinite(by["local_manual"].final_loss)
    assert math.isfinite(by["hf_manual"].final_loss)
    assert by["local_manual"].final_loss < 1e-2
    assert by["hf_manual"].final_loss < 1e-2


@pytest.mark.parametrize("seed", [0])
def test_scenario_a_report_relative_quality(seed: int):
    """
    Soft comparative check on matrix factorization (factored 2nd moment path).

    Does not require a universal winner; fails only if one side collapses
    while the other clearly solves the problem.
    """
    results = run_scenario_a_manual_lr(seed=seed, steps=400)["matrix_factor"]
    by = {r.name: r for r in results}
    local_f = by["local_manual"].final_loss
    hf_f = by["hf_manual"].final_loss
    assert math.isfinite(local_f) and math.isfinite(hf_f)
    # Neither should be >100x worse if the other reached a small loss.
    if min(local_f, hf_f) < 1e-3:
        assert max(local_f, hf_f) < 1e-1


def test_early_steps_diverge_due_to_beta2_schedule():
    """
    With identical grads/manual LR, trajectories diverge because HF uses
    beta2t=1-step**(-0.8) while local uses fixed beta2=0.99.

    Step-1 ||Δθ|| can match: update clipping forces RMS(Û)≤d, so with the same
    lr the first clipped step norms coincide even when V̂ differs. Divergence
    shows up as second-moment EMA paths separate.
    """
    m = run_early_update_divergence(seed=0)
    assert m["param_drift_step20"] > 1e-4
    # By step 10 the EMA states differ enough that update norms separate.
    assert m["upd_norm_step10_local"] != pytest.approx(
        m["upd_norm_step10_hf"], rel=1e-3, abs=1e-6
    )


def test_scenario_b_hf_relative_beats_badly_tuned_fixed_lr():
    """
    On matrix factorization, HF relative+scale should beat an untuned tiny fixed LR.
    """
    results = run_scenario_b_defaults(seed=0, steps=400)["matrix_factor"]
    by = {r.name: r for r in results}
    assert by["hf_relative_default"].final_loss < by["local_fixed_1e-3"].final_loss


def test_scenario_c_relative_flags_are_not_equivalent():
    """local relative_step=True is not HF relative_step=True; losses should differ."""
    results = run_scenario_c_relative_semantics(seed=0, steps=300)["matrix_factor"]
    by = {r.name: r for r in results}
    local_rel = by["local_relative_fork"].final_loss
    hf_rel = by["hf_relative"].final_loss
    assert math.isfinite(local_rel) and math.isfinite(hf_rel)
    # Different schedules → not numerically identical trajectories.
    assert not math.isclose(local_rel, hf_rel, rel_tol=1e-3, abs_tol=1e-6)


def test_full_benchmark_smoke_prints_summary():
    """End-to-end smoke: run all scenarios once and print a verdict helper."""
    a = run_scenario_a_manual_lr(seed=0, steps=250)
    b = run_scenario_b_defaults(seed=0, steps=250)
    c = run_scenario_c_relative_semantics(seed=0, steps=250)
    early = run_early_update_divergence(seed=0)

    print("\n=== Verdict helper (manual LR, matrix_factor) ===")
    mf = {r.name: r for r in a["matrix_factor"]}
    local_best = mf["local_manual"].best_loss
    hf_best = mf["hf_manual"].best_loss
    winner = "local" if local_best < hf_best else "hf"
    print(
        f"best_loss local={local_best:.3e} hf={hf_best:.3e} "
        f"winner={winner} ratio_local/hf={_ratio(local_best, hf_best):.3f}"
    )
    print(
        f"early step1 upd ratio local/hf="
        f"{early['upd_norm_step1_ratio_local_over_hf']:.3f} "
        f"drift20={early['param_drift_step20']:.3e}"
    )

    assert all(math.isfinite(r.final_loss) for r in a["quadratic_shifted"])
    assert all(math.isfinite(r.final_loss) for r in b["quadratic_shifted"])
    assert all(math.isfinite(r.final_loss) for r in c["quadratic_shifted"])


if __name__ == "__main__":
    test_full_benchmark_smoke_prints_summary()
    test_early_steps_diverge_due_to_beta2_schedule()
    test_scenario_a_both_converge_manual_lr(0)
    test_scenario_b_hf_relative_beats_badly_tuned_fixed_lr()
    test_scenario_c_relative_flags_are_not_equivalent()
    print("\nAll comparison checks passed.")
