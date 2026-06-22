"""
Diagnose per-param effective LR drop when grad shrinks (low loss).

User regime:
  lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0
  grad_rms ~ 1e-9 .. 1e-7
  update_rms ~ 1e-6 .. 1e-5

Effective LR proxies (per param):
  eff_explicit = _get_lr output
  eff_precond  = RMS(update_hat) / RMS(grad)
  eff_momentum = RMS(final_update) / RMS(scaled_update)  [~1 at steady state]
  eff_total    = RMS(final_update) / RMS(grad)
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testing"))
sys.path.insert(0, str(ROOT))

from simulate_adafactor_effective_lr import approx_sq_grad, get_lr, rms
from toolkit.optimizers.adafactor import Adafactor


USER = dict(
    lr=1e-4,
    min_lr=1e-6,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    beta2=0.99,
    beta1=0.9,
    scale_parameter=True,
    relative_step=True,
    rms_max=0.01,
    emergency_brake=None,
    instability_score=0.0,
    saddle_point_boost=1.0,
)


def make_state(shape, param_rms):
    return {
        "param": torch.full(shape, param_rms),
        "row": torch.zeros(shape[0]),
        "col": torch.zeros(shape[1]),
        "exp_avg": torch.zeros(shape),
    }


def step_and_decompose(grad, state, group):
    """One optimizer step with factorization (mirrors adafactor_step)."""
    beta2, beta1 = group["beta2"], group["beta1"]
    eps0, clip = group["eps"][0], group["clip_threshold"]

    p = state["param"]
    param_rms, grad_rms = rms(p), rms(grad)
    row, col = state["row"], state["col"]
    upd_sq = grad ** 2 + eps0
    row.mul_(beta2).add_(upd_sq.mean(-1), alpha=1 - beta2)
    col.mul_(beta2).add_(upd_sq.mean(-2), alpha=1 - beta2)

    pre = approx_sq_grad(row.clone(), col.clone()) * grad
    pre_rms = rms(pre)
    clip_d = max(pre_rms / clip, 1.0)
    update_hat = pre / clip_d

    exp_avg = state["exp_avg"]
    dc = F.cosine_similarity(update_hat.flatten(), exp_avg.flatten(), dim=0).item()
    lr = get_lr(param_rms, grad_rms, group, dc)
    scaled = update_hat * lr

    b1 = beta1 if beta1 is not None else 0.9
    exp_avg.mul_(b1).add_(scaled, alpha=1 - b1)
    final = exp_avg.clone() if beta1 is not None else scaled
    p.add_(-final)

    return {
        "grad_rms": grad_rms,
        "lr": lr,
        "clip_d": clip_d,
        "v_over_g2": row.mean().item() / (grad_rms ** 2 + 1e-60),
        "pre_rms": pre_rms,
        "hat_rms": rms(update_hat),
        "scaled_rms": rms(scaled),
        "final_rms": rms(final),
        "eff_explicit": lr,
        "eff_precond": rms(update_hat) / (grad_rms + 1e-30),
        "eff_after_lr": rms(scaled) / (grad_rms + 1e-30),
        "eff_momentum": rms(final) / (rms(scaled) + 1e-30),
        "eff_total": rms(final) / (grad_rms + 1e-30),
        "dir_consistency": dc,
        "clip_boost": 1.0 / clip_d,
    }


def sustained_decay_report(param_rms=0.001, warmup_g=1e-4, g0=1e-7, decay=0.92, steps=80):
    shape = (32, 64)
    d = torch.randn(shape)
    d = d / d.norm()
    state = make_state(shape, param_rms)
    group = dict(USER)

    for _ in range(300):
        step_and_decompose(d * warmup_g, state, group)

    print(f"\n=== sustained grad decay (param_rms={param_rms}, g0={g0}, decay={decay}) ===")
    print(
        "step   grad      lr        |final|   eff_tot  eff_prec  eff_mom  clip  v/g^2   dc"
    )
    g = g0
    rows = []
    for t in range(steps):
        m = step_and_decompose(d * g, state, group)
        rows.append(m)
        g *= decay
        if t % 8 == 0 or t == steps - 1:
            print(
                f"{t:4d}  {m['grad_rms']:.1e}  {m['lr']:.2e}  {m['final_rms']:.2e}  "
                f"{m['eff_total']:7.2f}  {m['eff_precond']:8.1f}  {m['eff_momentum']:6.3f}  "
                f"{m['clip_d']:4.2f}  {m['v_over_g2']:7.0f}  {m['dir_consistency']:.3f}"
            )
    return rows


def ablation_isolate(param_rms=0.001):
    """Which factor causes eff_total drop?"""
    shape = (32, 64)
    d = torch.randn(shape)
    d = d / d.norm()
    warmup_g, g_test = 1e-4, 5e-8

    configs = [
        ("baseline", {}),
        ("beta1=None", {"beta1": None}),
        ("beta2=0.9", {"beta2": 0.9}),
        ("no scale/relative", {"scale_parameter": False, "relative_step": False}),
    ]

    print("\n=== ablation after warmup (single step at grad=5e-8) ===")
    print("config                  eff_tot  |final|   eff_prec  lr        clip")
    for name, overrides in configs:
        group = dict(USER)
        group.update(overrides)
        state = make_state(shape, param_rms)
        for _ in range(300):
            step_and_decompose(d * warmup_g, state, group)
        m = step_and_decompose(d * g_test, state, group)
        print(
            f"{name:22s}  {m['eff_total']:7.2f}  {m['final_rms']:.2e}  "
            f"{m['eff_precond']:8.1f}  {m['lr']:.2e}  {m['clip_d']:4.2f}"
        )


def clip_regime_transition():
    """clip_threshold boost disappears when grad leaves clipped regime."""
    shape = (32, 64)
    d = torch.randn(shape)
    d = d / d.norm()
    group = dict(USER)
    group.update(scale_parameter=False, relative_step=False, beta1=None)

    state = make_state(shape, 0.001)
    for _ in range(200):
        step_and_decompose(d * 1e-4, state, group)

    print("\n=== clip regime: after warmup at grad=1e-4 ===")
    for g in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        st = {
            "param": state["param"].clone(),
            "row": state["row"].clone(),
            "col": state["col"].clone(),
            "exp_avg": torch.zeros(shape),
        }
        m = step_and_decompose(d * g, st, group)
        print(
            f"grad={g:.0e}  clip={m['clip_d']:.3f}  hat_rms={m['hat_rms']:.2e}  "
            f"eff_precond={m['eff_precond']:.2e}  boost={m['clip_boost']:.3f}"
        )


def real_adafactor_check():
    """Same scenario via production Adafactor.step()."""
    shape = (32, 64)
    p = torch.nn.Parameter(torch.full(shape, 0.001))
    opt = Adafactor(
        [p],
        lr=USER["lr"],
        beta1=USER["beta1"],
        beta2=USER["beta2"],
        scale_parameter=True,
        relative_step=True,
        weight_decay=0.0,
        min_lr=USER["min_lr"],
        eps=USER["eps"],
        clip_threshold=USER["clip_threshold"],
    )
    d = torch.randn(shape)
    d = d / d.norm()

    for _ in range(300):
        p.grad = (d * 1e-4).clone()
        opt.step()

    print("\n=== real Adafactor.step sustained decay ===")
    g = 1e-7
    prev = p.detach().clone()
    print("step   grad_rms  lr_mean   update_rms eff_tot")
    for t in range(60):
        p.grad = (d * g).clone()
        opt.step()
        gr = opt.state[p]["grad_rms"].item()
        upd = (prev - p.detach()).pow(2).mean().sqrt().item()
        prev = p.detach().clone()
        eff = upd / (gr + 1e-30)
        lr_g = opt.get_learning_rates()[0]
        g *= 0.92
        if t % 6 == 0 or t == 59:
            print(f"{t:4d}  {gr:.2e}  {lr_g:.2e}  {upd:.2e}  {eff:.2f}")


def plateau_vs_decay():
    """Stable tiny grad (plateau) vs sustained decay — opposite eff_total behavior."""
    shape = (32, 64)
    d = torch.randn(shape)
    d = d / d.norm()
    group = dict(USER)
    param_rms = 0.01  # lr_explicit = 1e-6

    state = make_state(shape, param_rms)
    for _ in range(300):
        step_and_decompose(d * 1e-4, state, group)

    print("\n=== plateau: stable grad after warmup (loss flat, grad tiny) ===")
    for g in [1e-7, 5e-8, 1e-8, 5e-9]:
        st = make_state(shape, param_rms)
        st["row"] = state["row"].clone()
        st["col"] = state["col"].clone()
        st["exp_avg"] = state["exp_avg"].clone()
        m = step_and_decompose(d * g, st, group)
        print(
            f"grad={g:.0e}  |final|={m['final_rms']:.2e}  eff_total={m['eff_total']:.1f}  "
            f"lr={m['lr']:.2e}"
        )

    print("\n=== decay: same config but grad *0.92 each step for 40 steps ===")
    st2 = make_state(shape, param_rms)
    for _ in range(300):
        step_and_decompose(d * 1e-4, st2, group)
    g = 1e-7
    for t in range(40):
        m = step_and_decompose(d * g, st2, group)
        g *= 0.92
    print(
        f"after decay: grad={m['grad_rms']:.2e}  |final|={m['final_rms']:.2e}  "
        f"eff_total={m['eff_total']:.1f}"
    )


def match_user_magnitudes():
    """Target: grad~1e-8, update~1e-6 (param_rms=0.01, warmup 1e-4)."""
    shape = (32, 64)
    d = torch.randn(shape)
    d = d / d.norm()
    group = dict(USER)
    param_rms = 0.01
    state = make_state(shape, param_rms)
    for _ in range(300):
        step_and_decompose(d * 1e-4, state, group)

    m = step_and_decompose(d * 1e-8, state, group)
    print("\n=== user magnitude check (param_rms=0.01, grad=1e-8) ===")
    print(f"grad_rms={m['grad_rms']:.2e}  |final|={m['final_rms']:.2e}  lr={m['lr']:.2e}")
    print(f"eff_total={m['eff_total']:.1f}  (update/grad ratio)")
    print(f"eff_precond={m['eff_precond']:.1f}  eff_momentum={m['eff_momentum']:.1f}")


def main():
    print("USER: lr=1e-4 beta1=0.9 beta2=0.99 wd=0")
    rows = sustained_decay_report(param_rms=0.001)
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        print("\n--- summary ---")
        print(f"eff_total:  {first['eff_total']:.2f} -> {last['eff_total']:.2f}")
        print(f"|final|:    {first['final_rms']:.2e} -> {last['final_rms']:.2e}")
        print(f"grad:       {first['grad_rms']:.2e} -> {last['grad_rms']:.2e}")
        print(f"explicit lr:{first['lr']:.2e} -> {last['lr']:.2e}  (flat expected)")
        print(f"eff_precond:{first['eff_precond']:.1f} -> {last['eff_precond']:.1f}")
        print(f"eff_momentum:{first['eff_momentum']:.3f} -> {last['eff_momentum']:.3f}")
        print(f"v/g^2:      {first['v_over_g2']:.0f} -> {last['v_over_g2']:.0f}")

    ablation_isolate()
    clip_regime_transition()
    plateau_vs_decay()
    match_user_magnitudes()
    real_adafactor_check()

    # param_rms sensitivity (LoRA scale)
    print("\n=== param_rms sensitivity (sustained decay endpoints) ===")
    for pr in [1e-3, 1e-2, 1e-1]:
        r = sustained_decay_report(param_rms=pr, steps=40)
        print(f"param_rms={pr}: eff {r[0]['eff_total']:.2f} -> {r[-1]['eff_total']:.2f}, lr={r[0]['lr']:.2e}")


if __name__ == "__main__":
    main()
