"""
Simulate per-parameter effective LR in Adafactor when gradient magnitude shrinks (low loss).

Effective LR definitions tracked:
  - lr: explicit _get_lr output
  - eff_precond: RMS(update_hat) / RMS(grad)  (preconditioner + clip, before LR)
  - eff_total:   RMS(final_update) / RMS(grad) (includes momentum if beta1 set)
"""
import math
import torch
import torch.nn.functional as F


def rms(t: torch.Tensor) -> float:
    return t.pow(2).mean().sqrt().item()


def approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt().unsqueeze(-1)
    c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
    return torch.mul(r_factor, c_factor)


def get_lr(param_rms, grad_rms, group, dir_consistency=None):
    """Mirror adafactor._get_lr (relative_step + scale_parameter path)."""
    base_lr = group["lr"]
    min_lr = group["min_lr"]
    eps0, eps1 = group["eps"]
    scale = 1.0
    relative = 1.0

    if group["scale_parameter"]:
        scale = max(eps1, param_rms)

    if group["relative_step"]:
        group_param_rms_max = group.get("rms_max", eps1)
        brake = 1.0
        soft_brake = 1.0
        eb = group.get("emergency_brake")
        if group["scale_parameter"] and eb is not None:
            dir_val = dir_consistency if dir_consistency is not None else 0.0
            brake = max(eb, min(1 + dir_val, 1.0))
            score = group.get("instability_score", 0.0)
            soft_brake = max(eb, math.exp(-score))
        ratio = max(eps0, (group_param_rms_max - param_rms) / (group_param_rms_max + eps0))
        relative = (1 + min_lr * ratio) * brake * soft_brake * group.get("saddle_point_boost", 1.0)

    new_lr = base_lr * scale * relative
    if group.get("emergency_brake") is not None:
        group_param_rms_max = group.get("rms_max", eps1)
        max_allowed = group_param_rms_max / 1000.0
        new_lr = min(new_lr, max_allowed)
    return new_lr


def adafactor_step(grad, state, group):
    beta2 = group["beta2"]
    beta1 = group["beta1"]
    clip = group["clip_threshold"]
    eps0 = group["eps"][0]
    factored = True

    p = state["param"]
    param_rms = rms(p)
    grad_rms = rms(grad)

    row, col = state["row"], state["col"]
    upd_sq = grad ** 2 + eps0
    row.mul_(beta2).add_(upd_sq.mean(dim=-1), alpha=1 - beta2)
    col.mul_(beta2).add_(upd_sq.mean(dim=-2), alpha=1 - beta2)

    pre = approx_sq_grad(row.clone(), col.clone()) * grad
    clip_denom = max(rms(pre) / clip, 1.0)
    update_hat = pre / clip_denom

    exp_avg = state["exp_avg"]
    dir_consistency = None
    if beta1 is not None:
        dir_consistency = F.cosine_similarity(update_hat.flatten(), exp_avg.flatten(), dim=0).item()

    lr = get_lr(param_rms, grad_rms, group, dir_consistency)
    scaled = update_hat * lr

    if beta1 is not None:
        exp_avg.mul_(beta1).add_(scaled, alpha=1 - beta1)
        final = exp_avg.clone()
    else:
        final = scaled

    p.add_(-final)
    return {
        "lr": lr,
        "grad_rms": grad_rms,
        "v_row_mean": row.mean().item(),
        "precond_rms": rms(pre),
        "update_hat_rms": rms(update_hat),
        "final_rms": rms(final),
        "eff_precond": rms(update_hat) / (grad_rms + 1e-30),
        "eff_total": rms(final) / (grad_rms + 1e-30),
        "dir_consistency": dir_consistency,
        "clip_denom": clip_denom,
    }


def run_scenario(name, grad_schedule, group_overrides=None):
    group = {
        "lr": 1e-3,
        "min_lr": 1e-6,
        "eps": (1e-30, 1e-3),
        "clip_threshold": 1.0,
        "beta2": 0.99,
        "beta1": 0.9,
        "scale_parameter": True,
        "relative_step": True,
        "rms_max": 0.05,
        "emergency_brake": None,
        "instability_score": 0.0,
        "saddle_point_boost": 1.0,
    }
    if group_overrides:
        group.update(group_overrides)

    shape = (8, 16)
    state = {
        "param": torch.full(shape, 0.02),
        "row": torch.zeros(shape[0]),
        "col": torch.zeros(shape[1]),
        "exp_avg": torch.zeros(shape),
    }

    direction = torch.randn(shape)
    direction = direction / direction.norm()

    print(f"\n=== {name} ===")
    print("step  grad_rms    lr        v_ema       clip_d   eff_precond  eff_total  dir_cons")
    for t, g_scale in enumerate(grad_schedule):
        grad = direction * g_scale
        m = adafactor_step(grad, state, group)
        if t % max(1, len(grad_schedule) // 12) == 0 or t == len(grad_schedule) - 1:
            dc = m["dir_consistency"]
            dc_s = f"{dc:7.4f}" if dc is not None else "    n/a"
            print(
                f"{t:4d}  {m['grad_rms']:.2e}  {m['lr']:.2e}  {m['v_row_mean']:.2e}  "
                f"{m['clip_denom']:6.3f}  {m['eff_precond']:10.4f}  {m['eff_total']:9.4f}  {dc_s}"
            )


def main():
    # Phase 1: large grads, phase 2: grad shrinks 100x (low loss / fine convergence)
    decay = [1e-2] * 80 + [1e-2 * (0.95 ** i) for i in range(120)]

    run_scenario("baseline (beta2=0.99, beta1=0.9)", decay)
    run_scenario("no momentum (beta1=None)", decay, {"beta1": None})
    run_scenario("fast v decay (beta2=0.9)", decay, {"beta2": 0.9})
    run_scenario("no relative_step", decay, {"relative_step": False})
    run_scenario("emergency_brake=0.2", decay, {"emergency_brake": 0.2})

    # isolate preconditioner: sudden grad drop after warm v_ema
    print("\n=== sudden grad drop (v_ema stale) ===")
    sched = [1e-2] * 50 + [1e-4] * 50 + [1e-5] * 50
    run_scenario("sudden drop beta2=0.99", sched)
    run_scenario("sudden drop beta2=0.9", sched, {"beta2": 0.9})


def compare_to_naive_lr():
    """Actual |update| vs |lr * grad| when grad decays."""
    group = {
        "lr": 1e-3,
        "min_lr": 1e-6,
        "eps": (1e-30, 1e-3),
        "clip_threshold": 1.0,
        "beta2": 0.99,
        "beta1": 0.9,
        "scale_parameter": True,
        "relative_step": True,
        "rms_max": 0.05,
        "emergency_brake": None,
        "instability_score": 0.0,
        "saddle_point_boost": 1.0,
    }
    shape = (8, 16)
    direction = torch.randn(shape)
    direction = direction / direction.norm()
    state = {
        "param": torch.full(shape, 0.02),
        "row": torch.zeros(shape[0]),
        "col": torch.zeros(shape[1]),
        "exp_avg": torch.zeros(shape),
    }
    sched = [1e-2] * 60 + [1e-2 * (0.95 ** i) for i in range(100)]

    print("\n=== |update| vs |lr * grad| (beta2 stale v) ===")
    print("step  grad_rms   lr       |update|   |lr*grad|  ratio   clip  v/g^2")
    for t, g_scale in enumerate(sched):
        grad = direction * g_scale
        m = adafactor_step(grad, state, group)
        lr = m["lr"]
        naive_rms = (lr * grad).pow(2).mean().sqrt().item()
        ratio = m["final_rms"] / (naive_rms + 1e-30)
        v_over_g2 = m["v_row_mean"] / (g_scale ** 2 + 1e-30)
        if t % 15 == 0 or t == len(sched) - 1:
            print(
                f"{t:3d}  {m['grad_rms']:.2e} {lr:.2e} {m['final_rms']:.2e} "
                f"{naive_rms:.2e} {ratio:6.4f} {m['clip_denom']:5.2f} {v_over_g2:8.1f}"
            )


def convergence_plateau():
    """Stable direction, grad shrinks; track dynamic_gain = |update|/|grad|."""
    group = {
        "lr": 1e-3,
        "min_lr": 1e-6,
        "eps": (1e-30, 1e-3),
        "clip_threshold": 1.0,
        "beta2": 0.99,
        "beta1": 0.9,
        "scale_parameter": True,
        "relative_step": True,
        "rms_max": 0.05,
        "emergency_brake": None,
        "saddle_point_boost": 1.0,
    }
    shape = (8, 16)
    d = torch.randn(shape)
    d = d / d.norm()
    state = {
        "param": torch.full(shape, 0.02),
        "row": torch.zeros(shape[0]),
        "col": torch.zeros(shape[1]),
        "exp_avg": torch.zeros(shape),
    }

    # warm-up with large grads + clip boost to build exp_avg and v_ema
    for _ in range(100):
        adafactor_step(d * 1e-2, state, group)

    print("\n=== plateau: grad decays, dir_consistency ~ 1 ===")
    print("step  grad_rms   lr       dyn_gain  |update|  clip  v/g^2")
    for t, g in enumerate([1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]):
        m = adafactor_step(d * g, state, group)
        dg = m["final_rms"] / (m["grad_rms"] + 1e-30)
        vg = m["v_row_mean"] / (g ** 2 + 1e-30)
        print(
            f"{t:3d}  {m['grad_rms']:.2e} {m['lr']:.2e} {dg:8.2f} "
            f"{m['final_rms']:.2e} {m['clip_denom']:5.2f} {vg:8.1f}"
        )


def old_activity_lr(grad_rms, grad_rms_max, param_rms, group_rms_max, cap_lr, eps0, eps1):
    """364bb1d formula: activity = grad_rms / grad_rms_max."""
    weight = max(eps1, (group_rms_max - param_rms) / (group_rms_max + eps0))
    activity = grad_rms / (grad_rms_max + eps0)
    protection = min(1.0, max(param_rms, eps0) / (grad_rms + eps0))
    return cap_lr * weight * activity * protection


def simulate_old_vs_new_lr():
    """Show explicit lr drop from removed grad_rms activity term."""
    cap_lr = 1e-3
    eps0, eps1 = 1e-30, 1e-3
    param_rms = 0.02
    group_rms_max = 0.05
    grad_rms_max = 1e-2  # stale running max after training

    print("\n=== explicit lr: old (grad_rms activity) vs current _get_lr ===")
    print("grad_rms   old_lr      current_lr   activity")
    for g in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        old = old_activity_lr(g, grad_rms_max, param_rms, group_rms_max, cap_lr, eps0, eps1)
        # current: scale * (1 + min_lr * ratio), no grad_rms
        min_lr = 1e-6
        scale = max(eps1, param_rms)
        ratio = max(eps0, (group_rms_max - param_rms) / (group_rms_max + eps0))
        current = cap_lr * scale * (1 + min_lr * ratio)
        act = g / (grad_rms_max + eps0)
        print(f"{g:.1e}  {old:.2e}  {current:.2e}  {act:.4f}")


def clip_boost_effect():
    """Isolate clip_threshold: boost disappears when grad shrinks."""
    group = {
        "lr": 1e-3, "min_lr": 1e-6, "eps": (1e-30, 1e-3), "clip_threshold": 1.0,
        "beta2": 0.99, "beta1": None, "scale_parameter": False, "relative_step": False,
    }
    shape = (8, 16)
    d = torch.randn(shape); d /= d.norm()
    state = {"param": torch.zeros(shape), "row": torch.zeros(shape[0]), "col": torch.zeros(shape[1]), "exp_avg": torch.zeros(shape)}

    # build v_ema at large grad
    for _ in range(200):
        adafactor_step(d * 1e-2, state, group)

    print("\n=== clip boost loss + stale v (beta1=None, no relative_step) ===")
    print("grad_rms  update_hat_rms  lr*update_hat  boost=hat_rms/(pre/rms_clip)")
    for g in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        row, col = state["row"].clone(), state["col"].clone()
        grad = d * g
        upd_sq = grad ** 2 + 1e-30
        row.mul_(0.99).add_(upd_sq.mean(-1), alpha=0.01)
        col.mul_(0.99).add_(upd_sq.mean(-2), alpha=0.01)
        pre = approx_sq_grad(row, col) * grad
        pre_rms = rms(pre)
        clip_d = max(pre_rms / 1.0, 1.0)
        hat_rms = pre_rms / clip_d
        print(f"{g:.1e}  {hat_rms:.2e}  {hat_rms*1e-3:.2e}  {clip_d:.2f}")


def sudden_grad_drop_stale_v():
    """After large-grad warmup, tiny grad: stale v_ema vs fresh."""
    group = {
        "lr": 1e-3, "min_lr": 1e-6, "eps": (1e-30, 1e-3), "clip_threshold": 1.0,
        "beta2": 0.99, "beta1": 0.9, "scale_parameter": True, "relative_step": True,
        "rms_max": 0.05, "emergency_brake": None, "saddle_point_boost": 1.0,
    }
    shape = (8, 16)
    d = torch.randn(shape)
    d = d / d.norm()
    state = {
        "param": torch.full(shape, 0.02),
        "row": torch.zeros(shape[0]),
        "col": torch.zeros(shape[1]),
        "exp_avg": torch.zeros(shape),
    }
    for _ in range(100):
        adafactor_step(d * 1e-2, state, group)

    m = adafactor_step(d * 1e-5, state, group)
    state2 = {
        "param": torch.full(shape, 0.02),
        "row": torch.zeros(shape[0]),
        "col": torch.zeros(shape[1]),
        "exp_avg": state["exp_avg"].clone(),
    }
    m2 = adafactor_step(d * 1e-5, state2, group)

    print("\n=== sudden grad drop: stale v_ema (beta2) vs fresh ===")
    print(f"stale: grad={m['grad_rms']:.2e} hat={m['update_hat_rms']:.2e} "
          f"final={m['final_rms']:.2e} gain={m['final_rms']/m['grad_rms']:.1f} lr={m['lr']:.2e}")
    print(f"fresh: grad={m2['grad_rms']:.2e} hat={m2['update_hat_rms']:.2e} "
          f"final={m2['final_rms']:.2e} gain={m2['final_rms']/m2['grad_rms']:.1f}")
    print(f"stale v/g^2={m['v_row_mean']/(1e-5**2):.0f}  (>>1 means v remembers large grads)")


def sustained_small_grad_momentum():
    """Many steps at tiny grad after warmup: does dyn_gain (eff LR proxy) fall?"""
    group = {
        "lr": 1e-3, "min_lr": 1e-6, "eps": (1e-30, 1e-3), "clip_threshold": 1.0,
        "beta2": 0.99, "beta1": 0.9, "scale_parameter": True, "relative_step": True,
        "rms_max": 0.05, "emergency_brake": None, "saddle_point_boost": 1.0,
    }
    shape = (8, 16)
    d = torch.randn(shape)
    d = d / d.norm()
    state = {
        "param": torch.full(shape, 0.02),
        "row": torch.zeros(shape[0]),
        "col": torch.zeros(shape[1]),
        "exp_avg": torch.zeros(shape),
    }
    for _ in range(100):
        adafactor_step(d * 1e-2, state, group)

    print("\n=== sustained small grad (beta2=0.99, beta1=0.9) ===")
    print("step  grad_rms   lr       dyn_gain  |final|   hat_rms  dir_cons")
    for t in range(50):
        g = 1e-5 * (0.98 ** t)
        m = adafactor_step(d * g, state, group)
        if t % 5 == 0:
            dg = m["final_rms"] / (m["grad_rms"] + 1e-30)
            dc = m["dir_consistency"] or 0.0
            print(
                f"{t:3d}  {m['grad_rms']:.2e} {m['lr']:.2e} {dg:8.2f} "
                f"{m['final_rms']:.2e} {m['update_hat_rms']:.2e} {dc:.4f}"
            )


if __name__ == "__main__":
    main()
    compare_to_naive_lr()
    convergence_plateau()
    simulate_old_vs_new_lr()
    clip_boost_effect()
    sudden_grad_drop_stale_v()
    sustained_small_grad_momentum()
