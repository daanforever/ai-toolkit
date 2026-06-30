"""
Simulate LoRA-like split regime:
- "composition" branch quickly cools down (small grads)
- "detail" branch keeps stronger grads

Compares static beta2 vs adaptive beta2 in production Adafactor.
"""

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from toolkit.optimizers.adafactor import Adafactor


def _norm_direction(shape, offset):
    v = torch.arange(offset, offset + shape[0] * shape[1], dtype=torch.float32).view(*shape)
    return v / v.norm()


def _run_scenario(name, **opt_kwargs):
    shape = (16, 32)
    p_comp = torch.nn.Parameter(torch.full(shape, 0.01))
    p_detail = torch.nn.Parameter(torch.full(shape, 0.01))
    optimizer = Adafactor(
        [p_comp, p_detail],
        lr=1e-4,
        beta1=0.9,
        scale_parameter=True,
        relative_step=True,
        weight_decay=0.0,
        clip_threshold=1.0,
        **opt_kwargs,
    )

    d_comp = _norm_direction(shape, 1)
    d_detail = _norm_direction(shape, 777)

    # Common warmup: both branches are active.
    for _ in range(100):
        p_comp.grad = (d_comp * 1e-3).clone()
        p_detail.grad = (d_detail * 1e-3).clone()
        optimizer.step()

    comp_updates = []
    detail_updates = []
    comp_beta2 = []
    detail_beta2 = []

    comp_prev = p_comp.detach().clone()
    detail_prev = p_detail.detach().clone()

    # Divergent phase: composition cools down, detail stays active.
    for t in range(120):
        comp_scale = 1e-3 * (0.95 ** t)
        detail_scale = 1e-3

        p_comp.grad = (d_comp * comp_scale).clone()
        p_detail.grad = (d_detail * detail_scale).clone()
        optimizer.step()

        comp_updates.append((comp_prev - p_comp.detach()).pow(2).mean().sqrt().item())
        detail_updates.append((detail_prev - p_detail.detach()).pow(2).mean().sqrt().item())
        comp_prev = p_comp.detach().clone()
        detail_prev = p_detail.detach().clone()

        comp_beta2.append(float(optimizer.state[p_comp].get("beta2_effective", opt_kwargs.get("beta2", 0.99))))
        detail_beta2.append(float(optimizer.state[p_detail].get("beta2_effective", opt_kwargs.get("beta2", 0.99))))

    late = slice(-40, None)
    comp_late = sum(comp_updates[late]) / len(comp_updates[late])
    detail_late = sum(detail_updates[late]) / len(detail_updates[late])
    comp_beta2_late = sum(comp_beta2[late]) / len(comp_beta2[late])
    detail_beta2_late = sum(detail_beta2[late]) / len(detail_beta2[late])

    print(f"\n=== {name} ===")
    print(f"late comp update rms   : {comp_late:.3e}")
    print(f"late detail update rms : {detail_late:.3e}")
    print(f"detail/comp ratio      : {(detail_late / (comp_late + 1e-30)):.2f}")
    print(f"late comp beta2_eff    : {comp_beta2_late:.4f}")
    print(f"late detail beta2_eff  : {detail_beta2_late:.4f}")

    return {
        "comp_late": comp_late,
        "detail_late": detail_late,
        "ratio": detail_late / (comp_late + 1e-30),
        "comp_beta2_late": comp_beta2_late,
        "detail_beta2_late": detail_beta2_late,
    }


def main():
    baseline = _run_scenario("static beta2=0.999", beta2=0.999)
    adaptive = _run_scenario(
        "adaptive beta2 (0.999 -> 0.9 on low activity)",
        beta2=0.999,
        beta2_adaptive=True,
        beta2_min=0.9,
    )
    lower_static = _run_scenario("static beta2=0.95", beta2=0.95)

    print("\n=== comparison (late phase) ===")
    print(f"baseline detail/comp ratio : {baseline['ratio']:.2f}")
    print(f"adaptive detail/comp ratio : {adaptive['ratio']:.2f}")
    print(f"static 0.95 detail/comp    : {lower_static['ratio']:.2f}")


if __name__ == "__main__":
    main()
