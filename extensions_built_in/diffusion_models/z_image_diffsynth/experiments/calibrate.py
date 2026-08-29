"""Window-Δ equivalence, exchange rates, and beta* interpolation."""

from __future__ import annotations

from typing import Any


def classify_equivalence(
    *,
    equiv_ratio: float,
    equiv_cosine: float,
    rel_tol: float = 0.25,
    cosine_min: float = 0.9,
) -> dict[str, Any]:
    """Headline status for Δ_lr_x4 vs Δ_beta2_lo."""
    equiv_ratio = float(equiv_ratio)
    equiv_cosine = float(equiv_cosine)
    rel_tol = float(rel_tol)
    cosine_min = float(cosine_min)
    ratio_ok = abs(equiv_ratio - 1.0) <= rel_tol
    cosine_ok = equiv_cosine >= cosine_min
    if ratio_ok and cosine_ok:
        status = "equivalent"
    elif ratio_ok or cosine_ok:
        status = "partial"
    else:
        status = "divergent"
    return {
        "status": status,
        "equiv_ratio": equiv_ratio,
        "equiv_cosine": equiv_cosine,
        "rel_tol": rel_tol,
        "cosine_min": cosine_min,
    }


def pick_beta_star(
    *,
    rms_lr_x4: float,
    rms_by_beta: dict[float, float],
) -> dict[str, Any]:
    """beta whose rms(Δ) / rms(Δ_lr_x4) is nearest 1."""
    rms_lr_x4 = float(rms_lr_x4)
    if not rms_by_beta:
        return {
            "beta_star": None,
            "r_star": None,
            "ratios": {},
            "reason": "empty_grid",
        }

    best_beta = None
    best_r = None
    best_err = float("inf")
    ratios_out: dict[str, float] = {}
    for beta, rms_b in rms_by_beta.items():
        rms_b = float(rms_b)
        if rms_lr_x4 > 1e-12:
            r = rms_b / rms_lr_x4
        elif abs(rms_b) <= 1e-12:
            r = 1.0
        else:
            r = float("inf")
        ratios_out[str(float(beta))] = r
        err = abs(r - 1.0)
        if err < best_err:
            best_err = err
            best_beta = float(beta)
            best_r = r

    return {
        "beta_star": best_beta,
        "r_star": best_r,
        "ratios": ratios_out,
    }


def exchange_rates(
    *,
    rms_continue: float,
    rms_lr_x4: float,
    rms_beta2_lo: float,
) -> dict[str, float]:
    """s_lr / s_b2 = rms(Δ_fork) / rms(Δ_continue); 0 if continue ~ 0."""
    rms_continue = float(rms_continue)
    if rms_continue <= 1e-12:
        return {"s_lr": 0.0, "s_b2": 0.0}
    return {
        "s_lr": float(rms_lr_x4) / rms_continue,
        "s_b2": float(rms_beta2_lo) / rms_continue,
    }


def stationary_v_diagnostic(
    *,
    ratios_vs_continue: dict[float, float],
    s_lr: float,
) -> bool:
    """True iff all beta2-at-lr_base ratios ≈ 1 and s_lr ≥ 2 (diagnostics only)."""
    if not ratios_vs_continue:
        return False
    all_near_one = all(abs(float(r) - 1.0) < 0.15 for r in ratios_vs_continue.values())
    return bool(all_near_one and float(s_lr) >= 2.0)
