"""
Per-param effective LR when grad shrinks (low loss), user config:
  lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0

Decomposes: explicit lr | preconditioner (stale v) | clip boost | momentum
"""
import math
import torch
import torch.nn.functional as F

from simulate_adafactor_effective_lr import adafactor_step, rms, approx_sq_grad, get_lr

USER_LR = 1e-4
USER_BETA1 = 0.9
USER_BETA2 = 0.99


def base_group(**overrides):
    g = {
        "lr": USER_LR,
        "eps": (1e-30, 1e-3),
        "clip_threshold": 1.0,
        "beta2": USER_BETA2,
        "beta1": USER_BETA1,
        "scale_parameter": True,
        "relative_step": True,
        "rms_max": 0.05,
    }
    g.update(overrides)
    return g


def make_state(shape=(64, 128), param_val=0.02):
    return {
        "param": torch.full(shape, param_val),
        "row": torch.zeros(shape[0]),
        "col": torch.zeros(shape[1]),
        "exp_avg": torch.zeros(shape),
    }


def warmup(state, direction, group, grad_scale=1e-2, steps=200):
    for _ in range(steps):
        adafactor_step(direction * grad_scale, state, group)


def decompose_step(grad, state, group):
    """Mirror adafactor_step but return intermediate factors."""
    beta2 = group["beta2"]
    beta1 = group["beta1"]
    clip = group["clip_threshold"]
    eps0 = group["eps"][0]

    p = state["param"]
    param_rms = rms(p)
    grad_rms = rms(grad)

    row, col = state["row"], state["col"]
    upd_sq = grad ** 2 + eps0
    row.mul_(beta2).add_(upd_sq.mean(dim=-1), alpha=1 - beta2)
    col.mul_(beta2).add_(upd_sq.mean(dim=-2), alpha=1 - beta2)

    pre = approx_sq_grad(row.clone(), col.clone()) * grad
    pre_rms = rms(pre)
    clip_denom = max(pre_rms / clip, 1.0)
    update_hat = pre / clip_denom

    exp_avg = state["exp_avg"]
    beta2_direction_ema = state.get("beta2_direction_ema")
    if beta2_direction_ema is None or beta2_direction_ema.shape != update_hat.shape:
        beta2_direction_ema = torch.zeros_like(update_hat)
        state["beta2_direction_ema"] = beta2_direction_ema
    dc = F.cosine_similarity(update_hat.flatten(), beta2_direction_ema.flatten(), dim=0).item()
    beta2_direction_ema.mul_(beta2).add_(update_hat, alpha=1 - beta2)
    lr = get_lr(param_rms, grad_rms, group, dc)
    scaled = update_hat * lr

    b1 = beta1 if beta1 is not None else 0.9
    exp_avg_before = exp_avg.clone()
    exp_avg.mul_(b1).add_(scaled, alpha=1 - b1)
    final = exp_avg.clone() if beta1 is not None else scaled

    naive = (lr * grad).pow(2).mean().sqrt().item()
    v_mean = row.mean().item()

    return {
        "grad_rms": grad_rms,
        "lr": lr,
        "pre_rms": pre_rms,
        "clip_denom": clip_denom,
        "update_hat_rms": rms(update_hat),
        "scaled_rms": rms(scaled),
        "final_rms": rms(final),
        "naive_rms": naive,
        "eff_precond": rms(update_hat) / (grad_rms + 1e-30),
        "eff_total": rms(final) / (grad_rms + 1e-30),
        "eff_vs_naive": rms(final) / (naive + 1e-30),
        "v_over_g2": v_mean / (grad_rms ** 2 + 1e-30),
        "dir_consistency": dc,
        "clip_boost": 1.0 / clip_denom,
    }


def print_decay_table(title, group, grad_scales):
    shape = (64, 128)
    d = torch.randn(shape)
    d = d / d.norm()
    state = make_state(shape)
    warmup(state, d, group)

    print(f"\n=== {title} ===")
    hdr = (
        "grad_rms   lr        eff_tot  eff_prec  vs_naive  clip  v/g^2  "
        "hat_rms  |final|   dc"
    )
    print(hdr)
    for g in grad_scales:
        m = decompose_step(d * g, state, group)
        print(
            f"{m['grad_rms']:.1e}  {m['lr']:.2e}  {m['eff_total']:7.2f}  "
            f"{m['eff_precond']:8.4f}  {m['eff_vs_naive']:8.4f}  "
            f"{m['clip_denom']:4.2f}  {m['v_over_g2']:5.0f}  "
            f"{m['update_hat_rms']:.2e}  {m['final_rms']:.2e}  {m['dir_consistency']:.3f}"
        )


def sustained_decay():
    """Grad decays every step after warmup — like loss going down."""
    group = base_group()
    shape = (64, 128)
    d = torch.randn(shape)
    d = d / d.norm()
    state = make_state(shape)
    warmup(state, d, group, grad_scale=1e-2, steps=300)

    print("\n=== sustained grad decay (user config) ===")
    print("step  grad_rms   lr       eff_total  eff_prec  clip  v/g^2  dyn_gain")
    g = 1e-2
    for t in range(250):
        m = decompose_step(d * g, state, group)
        g *= 0.985
        if t % 25 == 0 or t == 249:
            print(
                f"{t:4d}  {m['grad_rms']:.2e}  {m['lr']:.2e}  "
                f"{m['eff_total']:9.2f}  {m['eff_precond']:8.4f}  "
                f"{m['clip_denom']:4.2f}  {m['v_over_g2']:6.0f}  "
                f"{m['final_rms']/(m['grad_rms']+1e-30):.2f}"
            )


def isolate_stale_v():
    """Same tiny grad, stale v_ema vs reset v only."""
    group = base_group()
    shape = (64, 128)
    d = torch.randn(shape)
    d = d / d.norm()

    state_stale = make_state(shape)
    warmup(state_stale, d, group)
    m_stale = decompose_step(d * 1e-5, state_stale, group)

    state_fresh = make_state(shape)
    state_fresh["exp_avg"] = state_stale["exp_avg"].clone()
    # fresh v, same momentum
    m_fresh = decompose_step(d * 1e-5, state_fresh, group)

    print("\n=== isolate stale v_ema (grad=1e-5 after warmup) ===")
    for label, m in [("stale v", m_stale), ("fresh v", m_fresh)]:
        print(
            f"{label:8s}: eff_total={m['eff_total']:.2f}  eff_precond={m['eff_precond']:.4f}  "
            f"clip={m['clip_denom']:.2f}  v/g^2={m['v_over_g2']:.0f}  lr={m['lr']:.2e}"
        )


def isolate_momentum():
    """Tiny grad: with vs without beta1."""
    shape = (64, 128)
    d = torch.randn(shape)
    d = d / d.norm()
    grad_scales = [1e-2, 1e-4, 1e-6]

    for beta1 in [0.9, None]:
        group = base_group(beta1=beta1)
        state = make_state(shape)
        warmup(state, d, group)
        print(f"\n=== momentum isolate beta1={beta1} ===")
        for g in grad_scales:
            m = decompose_step(d * g, state, group)
            print(
                f"grad={g:.0e}  eff_total={m['eff_total']:.4f}  "
                f"eff_precond={m['eff_precond']:.4f}  |final|={m['final_rms']:.2e}"
            )


def explicit_lr_vs_grad():
    """_get_lr does not use grad_rms — lr should stay flat."""
    group = base_group()
    param_rms = 0.02
    print("\n=== explicit lr vs grad_rms (current _get_lr) ===")
    print("grad_rms   lr")
    for gr in [1e-2, 1e-4, 1e-6, 1e-8]:
        lr = get_lr(param_rms, gr, group, dir_consistency=0.99)
        print(f"{gr:.1e}  {lr:.6e}")


def old_activity_formula():
    """Historical grad_rms/grad_rms_max term (not in current _get_lr)."""
    cap = USER_LR
    eps0, eps1 = 1e-30, 1e-3
    param_rms = 0.02
    group_rms_max = 0.05
    grad_rms_max = 1e-2  # stale after training

    print("\n=== old activity=grad_rms/grad_rms_max (removed from _get_lr) ===")
    print("grad_rms   old_lr      current_lr")
    scale = max(eps1, param_rms)
    ratio = max(eps0, (group_rms_max - param_rms) / (group_rms_max + eps0))
    current = cap * scale * (1 + 1e-6 * ratio)
    for gr in [1e-2, 1e-4, 1e-6, 1e-8]:
        weight = max(eps1, (group_rms_max - param_rms) / (group_rms_max + eps0))
        activity = gr / (grad_rms_max + eps0)
        protection = min(1.0, max(param_rms, eps0) / (gr + eps0))
        old = cap * weight * activity * protection
        print(f"{gr:.1e}  {old:.6e}  {current:.6e}")


def main():
    scales = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    print_decay_table("user config", base_group(), scales)
    print_decay_table("beta2=0.9", base_group(beta2=0.9), scales)
    print_decay_table("beta1=None", base_group(beta1=None), scales)
    print_decay_table("no relative_step", base_group(relative_step=False), scales)
    sustained_decay()
    isolate_stale_v()
    isolate_momentum()
    explicit_lr_vs_grad()
    old_activity_formula()


if __name__ == "__main__":
    main()
