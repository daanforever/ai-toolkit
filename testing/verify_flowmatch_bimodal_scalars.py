"""
Compare TensorBoard-exported timestep scalars to expected FlowMatch + gaussian_bimodal behavior.

YAML reference:
  noise_scheduler: flowmatch
  gaussian_mean: 300, gaussian_std: 0.2
  gaussian_mean_2: 800, gaussian_std_2: 0.2

`gaussian_mean*` are slot indices 0..ntt-1. FlowMatch uses linspace(1000, 1, ntt),
so sampled *values* should cluster near schedule[300] and schedule[800], not near 300/800.

`timestep_weights/min_timestep` is min over the logged batch of *values* (not slot indices).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from extensions_built_in.sd_trainer.gaussian_timestep_weights import evaluate_gaussian_timestep_bimodal
from toolkit.timestep_sampler import TimestepSampler, allowed_slot_index_range


def flowmatch_schedule(ntt: int = 1000) -> torch.Tensor:
    return torch.linspace(1000, 1, ntt, dtype=torch.float32)


def analyze_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    key = "timestep_weights/min_timestep"
    if key not in data:
        keys = list(data.keys())
        raise SystemExit(f"Missing {key!r}. Keys: {keys}")
    vals = [entry["value"] for entry in data[key]]
    t = torch.tensor(vals, dtype=torch.float64)
    return {
        "n": len(vals),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "p05": float(t.quantile(0.05)),
        "p50": float(t.quantile(0.50)),
        "p95": float(t.quantile(0.95)),
        "min": float(t.min()),
        "max": float(t.max()),
        "frac_gt_600": float((t > 600).float().mean()),
        "frac_gt_800": float((t > 800).float().mean()),
    }


def simulate_min_timestep(
    batch_size: int,
    ntt: int,
    min_noise_steps: int,
    max_noise_steps: int,
    *,
    mu1: float = 300.0,
    sigma1: float = 0.2,
    mu2: float = 800.0,
    sigma2: float = 0.2,
    n_steps: int = 5000,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    schedule = flowmatch_schedule(ntt)
    allowed_start, allowed_end = allowed_slot_index_range(
        ntt, min_noise_steps, max_noise_steps
    )

    class TC:
        num_train_timesteps = ntt
        timestep_type = "sigmoid"
        gaussian_mean = mu1
        gaussian_std = sigma1
        gaussian_mean_2 = mu2
        gaussian_std_2 = sigma2

    class NS:
        timesteps = schedule

    sampler = TimestepSampler(TC(), NS())
    latents = torch.zeros(1, device=device)

    mins = []
    for _ in range(n_steps):
        r = sampler.sample(
            batch_size,
            latents,
            "gaussian_bimodal",
            min_noise_steps,
            max_noise_steps,
            ntt,
            device,
            0,
        )
        mins.append(float(r.timesteps.min().item()))

    t = torch.tensor(mins, dtype=torch.float64)
    return {
        "batch_size": batch_size,
        "min_noise_steps": min_noise_steps,
        "max_noise_steps": max_noise_steps,
        "allowed_slots": (allowed_start, allowed_end),
        "n": len(mins),
        "mean": float(t.mean()),
        "p05": float(t.quantile(0.05)),
        "p50": float(t.quantile(0.50)),
        "p95": float(t.quantile(0.95)),
        "frac_gt_600": float((t > 600).float().mean()),
        "frac_gt_800": float((t > 800).float().mean()),
    }


def main() -> None:
    json_path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else r"d:\Projects\DeepSeek\ai-toolkit\temp\tmp\all_scalars.json"
    )
    if not json_path.is_file():
        print(f"File not found: {json_path}")
        sys.exit(1)

    obs = analyze_json(json_path)
    ntt = 1000
    sched = flowmatch_schedule(ntt)
    v300 = float(sched[300].item())
    v800 = float(sched[800].item())
    print("=== FlowMatch schedule (linspace 1000 -> 1, ntt=1000) ===")
    print(f"  schedule[300] = {v300:.4f}   (peak 1 in slot space)")
    print(f"  schedule[800] = {v800:.4f}   (peak 2 in slot space)")
    print()
    print("=== Observed TensorBoard scalar: timestep_weights/min_timestep ===")
    for k, v in obs.items():
        print(f"  {k}: {v}")
    print()

    print("=== Monte Carlo: min_denoising_steps=0, max_denoising_steps=999 (config default) ===")
    for bs in (1, 2, 4, 8, 16):
        sim = simulate_min_timestep(bs, ntt, 0, 999, n_steps=8000)
        print(
            f"  batch_size={bs:2d}: allowed {sim['allowed_slots']}  "
            f"mean={sim['mean']:.1f}  p05={sim['p05']:.1f} p50={sim['p50']:.1f} p95={sim['p95']:.1f}  "
            f"P(min>600)={sim['frac_gt_600']:.3f} P(min>800)={sim['frac_gt_800']:.3f}"
        )

    print()
    print("=== If denoising window excludes the Gaussian peaks (example) ===")
    # only high slot indices -> both Gaussian peaks are outside the window
    sim_narrow = simulate_min_timestep(4, ntt, min_noise_steps=1, max_noise_steps=100, n_steps=8000)
    print(
        f"  batch=4, max_denoising_steps=100, min=1 -> allowed {sim_narrow['allowed_slots']}: "
        f"mean={sim_narrow['mean']:.1f} p05={sim_narrow['p05']:.1f} p95={sim_narrow['p95']:.1f} "
        f"P(min>800)={sim_narrow['frac_gt_800']:.3f}"
    )

    # Weight mass at slots 300 & 800 (sanity)
    w = evaluate_gaussian_timestep_bimodal(
        torch.arange(ntt, dtype=torch.float32),
        300.0,
        0.2,
        800.0,
        0.2,
        torch.device("cpu"),
        torch.float32,
        ntt,
    )
    am = int(w.argmax().item())
    print()
    print("=== Bimodal weight table (50/50 mix, then / max) ===")
    print(
        f"  argmax slot {am} (sched value {sched[am].item():.2f}) — can differ slightly from mean_2"
        f" due to overlap of the two truncated normals"
    )
    print(f"  weight at slot 300: {w[300].item():.6f}  (sched {sched[300].item():.2f})")
    print(f"  weight at slot 800: {w[800].item():.6f}  (sched {sched[800].item():.2f})")


if __name__ == "__main__":
    main()
