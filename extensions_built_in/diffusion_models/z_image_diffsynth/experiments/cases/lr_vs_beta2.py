"""lr_vs_beta2 case helpers."""

from __future__ import annotations

CASE_ID = "lr_vs_beta2"


def resolve_prefix_steps(case: dict) -> int:
    if case.get("prefix_steps") is not None:
        return int(case["prefix_steps"])
    from math import ceil

    beta2_hi = float(case.get("beta2_hi", 0.99))
    return int(ceil(1.0 / (1.0 - beta2_hi)))


def calib_fork_specs(case: dict) -> list[dict]:
    """Resume forks needed for magnitude calibration (1 measure step each)."""
    lr_base = float(case["lr_base"])
    lr_hi = float(case["lr_hi"])
    beta2_hi = float(case.get("beta2_hi", 0.99))
    beta2_lo = float(case.get("beta2_lo", 0.9))
    out = [
        {"id": "continue", "lr": lr_base, "beta2": beta2_hi},
        {"id": "lr_x4", "lr": lr_hi, "beta2": beta2_hi},
        {"id": f"beta2_{beta2_lo}", "lr": lr_base, "beta2": beta2_lo},
        {"id": "both", "lr": lr_hi, "beta2": beta2_lo},
    ]
    for b in case.get("calibrate", {}).get("grid") or []:
        b = float(b)
        if abs(b - beta2_hi) < 1e-12 or abs(b - beta2_lo) < 1e-12:
            continue
        out.append({"id": f"beta2_{b}", "lr": lr_base, "beta2": b})
    return out
