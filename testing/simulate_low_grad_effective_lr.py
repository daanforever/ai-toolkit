"""
Per-param effective LR when grad shrinks (low loss plateau).

User-reported regime:
  lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0
  grad_rms ~ 1e-9 .. 1e-7, update_rms ~ 1e-8 .. 1e-6 (order-of-magnitude)

Effective LR proxies:
  eff_explicit  = lr from _get_lr
  eff_precond   = RMS(update_hat) / RMS(grad)
  eff_total     = RMS(final_update) / RMS(grad)   [includes beta1 momentum]
  eff_naive     = RMS(lr * grad) / RMS(grad) = lr
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testing"))

from simulate_adafactor_effective_lr import adafactor_step, get_lr, rms  # noqa: E402


def make_group(**overrides):
    group = {
        "lr": 1e-4,
        "min_lr": 1e-6,
        "eps": (1e-30, 1e-3),
        "clip_threshold": 1.0,
        "beta2": 0.99,
        "beta1": 0.9,
        "scale_parameter": True,
        "relative_step": True,
        "rms_max": 0.01,
        "emergency_brake": None,
        "instability_score": 0.0,
        "saddle_point_boost": 1.0,
    }
    group.update(overrides)
    return group


def warmup_then_decay(
    group,
    param_rms=0.001,
    warmup_grad=1e-4,
    warmup_steps=300,
    grad_levels=None,
):
    shape = (16, 32)
    direction = torch.randn(shape)
    direction = direction / direction.norm()

    state = {
        "param": torch.full(shape, param_rms),
        "row": torch.zeros(shape[0]),
        "col": torch.zeros(shape[1]),
        "exp_avg": torch.zeros(shape),
    }

    for _ in range(warmup_steps):
        adafactor_step(direction * warmup_grad, state, group)

    if grad_levels is None:
        grad_levels = [1e-7, 5e-8, 1e-8, 5e-9, 1e-9]

    rows = []
    for g in grad_levels:
        m = adafactor_step(direction * g, state, group)
        v_over_g2 = m["v_row_mean"] / (g * g + 1e-60)
        rows.append(
            {
                "grad_rms": g,
                "lr": m["lr"],
                "clip_denom": m["clip_denom"],
                "v_over_g2": v_over_g2,
                "update_hat_rms": m["update_hat_rms"],
                "final_rms": m["final_rms"],
                "eff_precond": m["eff_precond"],
                "eff_total": m["eff_total"],
                "dir_consistency": m["dir_consistency"],
            }
        )
    return rows


def sustained_decay(group, param_rms=0.001, warmup_grad=1e-4, warmup_steps=300, decay_steps=80):
    shape = (16, 32)
    direction = torch.randn(shape)
    direction = direction / direction.norm()
    state = {
        "param": torch.full(shape, param_rms),
        "row": torch.zeros(shape[0]),
        "col": torch.zeros(shape[1]),
        "exp_avg": torch.zeros(shape),
    }
    for _ in range(warmup_steps):
        adafactor_step(direction * warmup_grad, state, group)

    history = []
    g0 = 1e-7
    for t in range(decay_steps):
        g = g0 * (0.92 ** t)
        m = adafactor_step(direction * g, state, group)
        history.append(
            {
                "t": t,
                "grad_rms": g,
                "lr": m["lr"],
                "eff_total": m["eff_total"],
                "eff_precond": m["eff_precond"],
                "final_rms": m["final_rms"],
                "clip_denom": m["clip_denom"],
                "v_over_g2": m["v_row_mean"] / (g * g + 1e-60),
            }
        )
    return history


def decompose_one_step(group, grad_scale, state):
    """Break eff_total into lr * precond * momentum factors."""
    beta2 = group["beta2"]
    beta1 = group["beta1"]
    eps0 = group["eps"][0]
    clip = group["clip_threshold"]

    p = state["param"]
    grad = state["_last_grad"]
    param_rms = rms(p)
    grad_rms = rms(grad)

    row, col = state["row"], state["col"]
    upd_sq = grad ** 2 + eps0
    row_c, col_c = row.clone(), col.clone()
    row_c.mul_(beta2).add_(upd_sq.mean(dim=-1), alpha=1 - beta2)
    col_c.mul_(beta2).add_(upd_sq.mean(dim=-2), alpha=1 - beta2)

    from simulate_adafactor_effective_lr import approx_sq_grad

    pre = approx_sq_grad(row_c, col_c) * grad
    pre_rms = rms(pre)
    clip_d = max(pre_rms / clip, 1.0)
    update_hat = pre / clip_d

    beta2_direction_ema_before = state["beta2_direction_ema_before"]
    dir_consistency = F.cosine_similarity(
        update_hat.flatten(), beta2_direction_ema_before.flatten(), dim=0
    ).item()
    lr = get_lr(param_rms, grad_rms, group, dir_consistency)
    scaled = update_hat * lr

    exp_avg_after = exp_avg_before.clone()
    exp_avg_after.mul_(beta1).add_(scaled, alpha=1 - beta1)
    final = exp_avg_after

    return {
        "grad_rms": grad_rms,
        "lr": lr,
        "pre_rms": pre_rms,
        "clip_d": clip_d,
        "hat_rms": rms(update_hat),
        "scaled_rms": rms(scaled),
        "final_rms": rms(final),
        "eff_precond": rms(update_hat) / grad_rms,
        "eff_after_lr": rms(scaled) / grad_rms,
        "eff_total": rms(final) / grad_rms,
        "v_over_g2": row.mean().item() / (grad_scale ** 2 + 1e-60),
    }


def run_with_tracked_state(group, grad_levels, **warmup_kw):
    shape = (16, 32)
    direction = torch.randn(shape)
    direction = direction / direction.norm()
    state = {
        "param": torch.full(shape, warmup_kw.get("param_rms", 0.001)),
        "row": torch.zeros(shape[0]),
        "col": torch.zeros(shape[1]),
        "exp_avg": torch.zeros(shape),
    }
    for _ in range(warmup_kw.get("warmup_steps", 300)):
        adafactor_step(direction * warmup_kw.get("warmup_grad", 1e-4), state, group)

    out = []
    for g in grad_levels:
        grad = direction * g
        state["exp_avg_before"] = state["exp_avg"].clone()
        state["beta2_direction_ema_before"] = state.get("beta2_direction_ema", torch.zeros_like(grad)).clone()
        state["_last_grad"] = grad
        adafactor_step(grad, state, group)
        d = decompose_one_step(group, g, state)
        out.append(d)
    return out


def print_table(title, rows, keys):
    print(f"\n=== {title} ===")
    header = " ".join(f"{k:>14}" for k in keys)
    print(header)
    for r in rows:
        print(" ".join(f"{r[k]:14.4e}" if isinstance(r[k], float) else f"{str(r[k]):>14}" for k in keys))


def main():
    grad_levels = [1e-7, 5e-8, 1e-8, 5e-9, 1e-9]

    configs = [
        ("baseline scale+relative", make_group()),
        ("no relative_step", make_group(relative_step=False)),
        ("no scale_parameter", make_group(scale_parameter=False, relative_step=False)),
        ("beta1=None", make_group(beta1=None)),
        ("beta2=0.9", make_group(beta2=0.9)),
        ("emergency_brake=0.2", make_group(emergency_brake=0.2)),
    ]

    for name, group in configs:
        rows = warmup_then_decay(group, grad_levels=grad_levels)
        print_table(
            name,
            rows,
            ["grad_rms", "lr", "clip_denom", "v_over_g2", "eff_precond", "eff_total", "final_rms"],
        )

    print("\n=== sustained grad decay (baseline) ===")
    hist = sustained_decay(make_group())
    keys = ["t", "grad_rms", "lr", "eff_total", "eff_precond", "clip_denom", "v_over_g2"]
    print(" ".join(f"{k:>12}" for k in keys))
    for r in hist[::8] + [hist[-1]]:
        print(" ".join(f"{r[k]:12.4e}" for k in keys))

    print("\n=== factor decomposition at plateau ===")
    decomp = run_with_tracked_state(make_group(), grad_levels)
    print_table(
        "lr vs precond vs momentum",
        decomp,
        ["grad_rms", "lr", "clip_d", "eff_precond", "eff_after_lr", "eff_total", "v_over_g2"],
    )

    # clip boost: compare large-grad clip regime vs tiny grad
    print("\n=== clip_threshold boost: large grad vs tiny (same v_ema) ===")
    g = make_group(scale_parameter=False, relative_step=False, beta1=None)
    shape = (16, 32)
    d = torch.randn(shape)
    d = d / d.norm()
    for label, warmup_g, test_g in [
        ("warm large / test tiny", 1e-4, 1e-8),
        ("warm large / test large", 1e-4, 1e-4),
        ("warm tiny / test tiny", 1e-8, 1e-8),
    ]:
        st = {"param": torch.zeros(shape), "row": torch.zeros(shape[0]), "col": torch.zeros(shape[1]), "exp_avg": torch.zeros(shape)}
        for _ in range(200):
            adafactor_step(d * warmup_g, st, g)
        m = adafactor_step(d * test_g, st, g)
        print(
            f"{label:28s} grad={test_g:.1e} clip={m['clip_denom']:.3f} "
            f"eff_precond={m['eff_precond']:.2e} hat={m['update_hat_rms']:.2e}"
        )


if __name__ == "__main__":
    main()
