"""
Compare TensorBoard-exported timestep scalars to expected FlowMatch + gaussian_bimodal behavior.

Training pipeline (same as SDTrainer + TimestepSampler + log_timestep_weights):
  1) Sample integer slot indices in [allowed_lo, allowed_hi] from gaussian_bimodal weights
     (YAML gaussian_mean / gaussian_mean_2 are means in *index* space, 0..ntt-1).
  2) timesteps = noise_scheduler.timesteps[indices]  — FlowMatch: linspace(1000, 1, ntt).
  3) TensorBoard scalar timestep_weights/min_timestep = timesteps.min() over the batch.

JSON therefore stores scheduler *values*, not slot indices. To compare with YAML peaks at
300 and 850, map each logged value v to nearest i with schedule[i] ≈ v and histogram i.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Ensure imports like `extensions_built_in/...` work when running from any CWD.
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.sd_trainer.gaussian_timestep_weights import evaluate_gaussian_timestep_bimodal
from toolkit.timestep_sampler import TimestepSampler, allowed_slot_index_range


def flowmatch_schedule(ntt: int = 1000) -> torch.Tensor:
    return torch.linspace(1000, 1, ntt, dtype=torch.float32)


def nearest_slot_for_schedule_values(
    schedule: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """For each scalar v, index i minimizing |schedule[i] - v| (same discrete grid as the trainer)."""
    s = schedule.to(dtype=torch.float64).view(-1)
    v = values.to(dtype=torch.float64).view(-1, 1)
    return (v - s.view(1, -1)).abs().argmin(dim=1).to(torch.int64)


def analyze_json(path: Path) -> tuple[dict, torch.Tensor, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    key = "timestep_weights/min_timestep"
    if key not in data:
        keys = list(data.keys())
        raise SystemExit(f"Missing {key!r}. Keys: {keys}")
    raw_entries = list(data[key])
    vals = [entry["value"] for entry in raw_entries]
    t = torch.tensor(vals, dtype=torch.float64)
    stats = {
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
        "frac_150_250": float(((t >= 150) & (t <= 250)).float().mean()),
        "frac_650_750": float(((t >= 650) & (t <= 750)).float().mean()),
    }
    return stats, t, raw_entries


def histogram_by_100_timesteps(
    t: torch.Tensor,
    *,
    t_min: float = 0.0,
    t_max: float = 1000.0,
    bucket_size: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str], float]:
    """
    Histogram for timestep *values* into fixed-width buckets (default width=100).

    Buckets are intervals [t, t+bucket_size) and are printed as e.g. "0-99", "100-199".
    """
    if t.numel() == 0:
        bucket_count = int((t_max - t_min) / bucket_size)
        empty = torch.zeros((bucket_count,), dtype=torch.int64)
        labels = [f"{int(t_min + i * bucket_size)}-{int(t_min + (i + 1) * bucket_size - 1)}" for i in range(bucket_count)]
        return empty, empty.to(torch.float64), labels, float("nan")

    t = t.to(torch.float64).view(-1)
    bucket_count = int((t_max - t_min) / bucket_size)
    edges = torch.linspace(t_min, t_max, bucket_count + 1, dtype=t.dtype)
    # With right=False, edge is inclusive on the lower side:
    # bucket i covers [edges[i], edges[i+1]).
    bucket_idx = torch.bucketize(t, edges[1:-1], right=False)
    counts = torch.bincount(bucket_idx, minlength=bucket_count).to(torch.int64)
    perc = (counts.to(torch.float64) / float(t.numel())) * 100.0

    labels = [
        f"{int(edges[i].item())}-{int(edges[i + 1].item() - 1)}" for i in range(bucket_count)
    ]
    peak_bucket = int(torch.argmax(counts).item())
    peak_perc = float(perc[peak_bucket].item())
    return counts, perc, labels, peak_perc


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
    min_slots = []
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
        assert r.timestep_indices is not None
        pos = int(torch.argmin(r.timesteps).item())
        mins.append(float(r.timesteps[pos].item()))
        min_slots.append(int(r.timestep_indices[pos].item()))

    t = torch.tensor(mins, dtype=torch.float64)
    t_slots = torch.tensor(min_slots, dtype=torch.float64)
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
        "frac_150_250": float(((t >= 150) & (t <= 250)).float().mean()),
        "frac_650_750": float(((t >= 650) & (t <= 750)).float().mean()),
        "slot_mean": float(t_slots.mean()),
        "slot_p50": float(t_slots.quantile(0.50)),
        "frac_slot_250_350": float(((t_slots >= 250) & (t_slots <= 350)).float().mean()),
        "frac_slot_800_900": float(((t_slots >= 800) & (t_slots <= 900)).float().mean()),
        "_tensor": t,
        "_slot_tensor": t_slots.to(torch.int64),
    }


def _bootstrap_stat(
    ref: torch.Tensor,
    n_obs: int,
    stat_fn,
    *,
    n_rounds: int,
    seed: int,
    q_low: float = 0.005,
    q_high: float = 0.995,
) -> tuple[float, float, float]:
    """Return (obs would be passed outside) — actually returns (low, high) envelope for stat."""
    g = torch.Generator()
    g.manual_seed(seed)
    m = ref.shape[0]
    stats = []
    for _ in range(n_rounds):
        idx = torch.randint(0, m, (n_obs,), generator=g)
        stats.append(stat_fn(ref[idx]))
    st = torch.tensor(stats, dtype=torch.float64)
    return (
        float(st.quantile(q_low)),
        float(st.quantile(q_high)),
        float(st.median()),
    )


def run_checks(
    obs_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    *,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    alpha_tag: str,
) -> list[str]:
    n_obs = int(obs_tensor.shape[0])
    failures: list[str] = []

    def stat_mean(x: torch.Tensor) -> float:
        return float(x.mean())

    def stat_p50(x: torch.Tensor) -> float:
        return float(x.quantile(0.50))

    def stat_frac_gt_600(x: torch.Tensor) -> float:
        return float((x > 600).float().mean())

    def stat_frac_gt_800(x: torch.Tensor) -> float:
        return float((x > 800).float().mean())

    def stat_frac_150_250(x: torch.Tensor) -> float:
        return float(((x >= 150) & (x <= 250)).float().mean())

    def stat_frac_650_750(x: torch.Tensor) -> float:
        return float(((x >= 650) & (x <= 750)).float().mean())

    checks = [
        ("mean", stat_mean, float(obs_tensor.mean())),
        ("p50", stat_p50, float(obs_tensor.quantile(0.50))),
        ("frac_gt_600", stat_frac_gt_600, stat_frac_gt_600(obs_tensor)),
        ("frac_gt_800", stat_frac_gt_800, stat_frac_gt_800(obs_tensor)),
        ("frac_150_250 (near sched[800]≈200)", stat_frac_150_250, stat_frac_150_250(obs_tensor)),
        ("frac_650_750 (near sched[300]≈700)", stat_frac_650_750, stat_frac_650_750(obs_tensor)),
    ]

    for name, fn, observed in checks:
        lo, hi, med = _bootstrap_stat(
            ref_tensor,
            n_obs,
            fn,
            n_rounds=bootstrap_rounds,
            seed=bootstrap_seed,
        )
        if not (lo <= observed <= hi):
            failures.append(
                f"[{alpha_tag}] {name}: observed={observed:.6g} "
                f"outside bootstrap [{lo:.6g}, {hi:.6g}] (median={med:.6g}, n_obs={n_obs})"
            )
    return failures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "json_path",
        nargs="?",
        default=r"d:\Projects\DeepSeek\ai-toolkit\temp\tmp\all_scalars.json",
        type=Path,
        help="TensorBoard-exported scalars JSON",
    )
    p.add_argument("--ntt", type=int, default=1000, help="num_train_timesteps / schedule length")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--min-noise-steps", type=int, default=5, help="YAML train.min_denoising_steps")
    p.add_argument("--max-noise-steps", type=int, default=995, help="YAML train.max_denoising_steps")
    p.add_argument("--ref-steps", type=int, default=80_000, help="MC draws for reference pool")
    p.add_argument("--ref-seed", type=int, default=42)
    p.add_argument("--bootstrap-rounds", type=int, default=4000)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument(
        "--skip-checks",
        action="store_true",
        help="Only print diagnostics, do not compare JSON to MC",
    )
    p.add_argument(
        "--no-exit-on-fail",
        action="store_true",
        help="Print failures but exit 0",
    )
    p.add_argument("--gaussian-mean", type=float, default=300.0)
    p.add_argument("--gaussian-std", type=float, default=0.2)
    p.add_argument("--gaussian-mean-2", type=float, default=800.0)
    p.add_argument("--gaussian-std-2", type=float, default=0.2)
    p.add_argument(
        "--preview-rows",
        type=int,
        default=16,
        help="Print first N JSON rows: step, logged scheduler value, nearest slot index",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    json_path = args.json_path
    if not json_path.is_file():
        print(f"File not found: {json_path}")
        sys.exit(1)

    obs, obs_tensor, json_rows = analyze_json(json_path)
    ntt = args.ntt
    sched = flowmatch_schedule(ntt)
    obs_slots = nearest_slot_for_schedule_values(sched, obs_tensor).to(torch.float64)
    mu1_slot = int(args.gaussian_mean)
    mu2_slot = int(args.gaussian_mean_2)
    mu1_slot = max(0, min(mu1_slot, ntt - 1))
    mu2_slot = max(0, min(mu2_slot, ntt - 1))
    v_mu1 = float(sched[mu1_slot].item())
    v_mu2 = float(sched[mu2_slot].item())
    print("=== What training samples vs what JSON contains ===")
    print(
        "  Sampler draws slot indices i in [allowed_lo, hi], then "
        "timesteps = noise_scheduler.timesteps[i] (FlowMatch linspace 1000→1)."
    )
    print(
        "  TensorBoard timestep_weights/min_timestep = min(timesteps) on the batch "
        "(toolkit/util/tensorboard_timestep_weights.py)."
    )
    print(
        f"  YAML gaussian_mean / gaussian_mean_2 are means in *slot index* space "
        f"(here ~{mu1_slot} and ~{mu2_slot}), not the numbers stored in JSON."
    )
    print()
    print("=== FlowMatch schedule (linspace 1000 -> 1, ntt=%d) ===" % ntt)
    print(f"  slot {mu1_slot:4d}  ->  scheduler value {v_mu1:.4f}   (mode 1 after index→value map)")
    print(f"  slot {mu2_slot:4d}  ->  scheduler value {v_mu2:.4f}   (mode 2 after index→value map)")
    print()
    print("=== Observed TensorBoard scalar: timestep_weights/min_timestep ===")
    for k, v in obs.items():
        print(f"  {k}: {v}")
    print()
    print(
        "=== Inferred slot index per JSON point (nearest i: |schedule[i] - logged_value| min) ==="
    )
    print(
        f"  (batch_size=1: this is the training slot; batch>1: slot of the sample that had batch min value)"
    )
    print(f"  n: {int(obs_slots.numel())}  mean: {float(obs_slots.mean()):.2f}  "
          f"p50: {float(obs_slots.quantile(0.5)):.1f}  "
          f"P(slot in [250,350]): {float(((obs_slots >= 250) & (obs_slots <= 350)).float().mean()):.4f}  "
          f"P(slot in [800,900]): {float(((obs_slots >= 800) & (obs_slots <= 900)).float().mean()):.4f}")
    print()
    pr = max(0, args.preview_rows)
    if pr > 0:
        print(f"=== JSON preview (first {pr} rows): global_step, logged value, inferred slot ===")
        for row in json_rows[:pr]:
            st = row.get("step", "")
            val = float(row["value"])
            si = int(
                nearest_slot_for_schedule_values(
                    sched, torch.tensor([val], dtype=torch.float64)
                )[0].item()
            )
            print(f"  step {st:6}  logged={val:8.2f}  nearest_slot={si:4d}  sched[{si}]={float(sched[si].item()):.4f}")
        print()

    hist_counts, hist_perc, labels, peak_perc = histogram_by_100_timesteps(obs_tensor)
    print("=== Histogram: logged scheduler values (same units as JSON / TensorBoard) ===")
    print(f"  peak bucket count={int(hist_counts.max().item())} ({peak_perc:.2f}%)")
    for label, p in zip(labels, hist_perc.tolist()):
        print(f"  {label} {p:.2f}%")
    print()

    sh_counts, sh_perc, slabels, speak = histogram_by_100_timesteps(obs_slots)
    print("=== Histogram: inferred slot indices (compare to YAML gaussian_mean / gaussian_mean_2) ===")
    print(f"  peak bucket count={int(sh_counts.max().item())} ({speak:.2f}%)")
    for label, p in zip(slabels, sh_perc.tolist()):
        print(f"  {label} {p:.2f}%")
    print()

    sim_ref = simulate_min_timestep(
        args.batch_size,
        ntt,
        args.min_noise_steps,
        args.max_noise_steps,
        mu1=args.gaussian_mean,
        sigma1=args.gaussian_std,
        mu2=args.gaussian_mean_2,
        sigma2=args.gaussian_std_2,
        n_steps=args.ref_steps,
        seed=args.ref_seed,
    )
    ref_tensor = sim_ref.pop("_tensor")
    ref_slot_tensor = sim_ref.pop("_slot_tensor")
    print(
        f"=== Reference MC (batch={args.batch_size}, "
        f"min_denoising={args.min_noise_steps}, max_denoising={args.max_noise_steps}, "
        f"n={args.ref_steps}, seed={args.ref_seed}) ==="
    )
    print(f"  allowed_slots: {sim_ref['allowed_slots']}")
    for k in (
        "mean",
        "p05",
        "p50",
        "p95",
        "frac_gt_600",
        "frac_gt_800",
        "frac_150_250",
        "frac_650_750",
        "slot_mean",
        "slot_p50",
        "frac_slot_250_350",
        "frac_slot_800_900",
    ):
        print(f"  {k}: {sim_ref[k]}")
    rh_counts, rh_perc, rlabels, rpeak = histogram_by_100_timesteps(
        ref_slot_tensor.to(torch.float64)
    )
    print("  slot histogram (bucket size=100 on index axis):")
    print(f"    peak {int(rh_counts.max().item())} ({rpeak:.2f}%)")
    for label, p in zip(rlabels, rh_perc.tolist()):
        print(f"    {label} {p:.2f}%")
    print()

    print("=== Reference table: other batch sizes (min=0, max=999) ===")
    for bs in (1, 2, 4, 8, 16):
        sim = simulate_min_timestep(bs, ntt, 0, 999, n_steps=8000, seed=args.ref_seed + bs)
        sim.pop("_tensor", None)
        print(
            f"  batch_size={bs:2d}: allowed {sim['allowed_slots']}  "
            f"mean={sim['mean']:.1f}  p05={sim['p05']:.1f} p50={sim['p50']:.1f} p95={sim['p95']:.1f}  "
            f"P(min>600)={sim['frac_gt_600']:.3f} P(min>800)={sim['frac_gt_800']:.3f}"
        )

    print()
    print("=== Narrow window example (batch=4, min=1, max=100) ===")
    sim_narrow = simulate_min_timestep(4, ntt, min_noise_steps=1, max_noise_steps=100, n_steps=8000)
    sim_narrow.pop("_tensor", None)
    print(
        f"  allowed {sim_narrow['allowed_slots']}: "
        f"mean={sim_narrow['mean']:.1f} p05={sim_narrow['p05']:.1f} p95={sim_narrow['p95']:.1f} "
        f"P(min>800)={sim_narrow['frac_gt_800']:.3f}"
    )

    w = evaluate_gaussian_timestep_bimodal(
        torch.arange(ntt, dtype=torch.float32),
        args.gaussian_mean,
        args.gaussian_std,
        args.gaussian_mean_2,
        args.gaussian_std_2,
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
    print(
        f"  weight at slot {mu1_slot}: {w[mu1_slot].item():.6f}  (sched {sched[mu1_slot].item():.2f})"
    )
    print(
        f"  weight at slot {mu2_slot}: {w[mu2_slot].item():.6f}  (sched {sched[mu2_slot].item():.2f})"
    )

    if args.skip_checks:
        return

    print()
    print(
        f"=== Checks: JSON vs reference (bootstrap n={args.bootstrap_rounds}, "
        f"99% envelope on i.i.d. subsamples of size {obs['n']}) ==="
    )
    failures = run_checks(
        obs_tensor,
        ref_tensor,
        bootstrap_rounds=args.bootstrap_rounds,
        bootstrap_seed=args.bootstrap_seed,
        alpha_tag="0.5%..99.5%",
    )
    if failures:
        for line in failures:
            print(f"FAIL: {line}")
        if not args.no_exit_on_fail:
            sys.exit(1)
    else:
        print("OK: all scalar statistics fall inside bootstrap envelope.")


if __name__ == "__main__":
    main()
