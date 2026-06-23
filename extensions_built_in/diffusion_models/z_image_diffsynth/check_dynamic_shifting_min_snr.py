"""
Compatibility check: use_dynamic_shifting (true/false) x min_snr_gamma (0/1/3/5).

Run from repo root:
  venv\\Scripts\\python.exe extensions_built_in/diffusion_models/z_image_diffsynth/check_dynamic_shifting_min_snr.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from extensions_built_in.diffusion_models.z_image_diffsynth.scheduler_config import build_scheduler_config
from extensions_built_in.diffusion_models.z_image_diffsynth.snr_weighting_checks import (
    assert_apply_snr_flow_match_weights,
    expected_flow_match_min_snr_weight,
    lookup_snr,
)
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.train_tools import apply_snr_weight


GAMMAS = (0.0, 1.0, 3.0, 5.0)
# (latent_h, latent_w, pixel_label) — Z-Image / Flux VAE is 8x: 512→64, 1024→128
RESOLUTIONS = ((64, 64, 512), (128, 128, 1024))
SAMPLE_TS = [1.0, 10.0, 50.0, 200.0, 500.0, 800.0, 990.0, 1000.0]


@dataclass
class ScheduleStats:
    use_dynamic_shifting: bool
    gamma: float
    h: int
    w: int
    pixel: int
    n_timesteps: int
    t_min: float
    t_max: float
    t_mean: float
    frac_capped_on_schedule: float
    weight_min: float
    weight_max: float
    has_nan: bool
    has_inf: bool
    formula_ok: bool


def make_scheduler(use_dynamic_shifting: bool) -> CustomFlowMatchEulerDiscreteScheduler:
    return CustomFlowMatchEulerDiscreteScheduler(**build_scheduler_config(use_dynamic_shifting))


def build_shift_schedule(sched, h: int, w: int) -> torch.Tensor:
    latents = torch.zeros(1, 16, h, w)
    sched.set_train_timesteps(
        1000,
        device="cpu",
        timestep_type="shift",
        latents=latents,
    )
    return sched.timesteps.clone()


def cap_fraction_on_schedule(timesteps: torch.Tensor, gamma: float) -> float:
    ntt = 1000.0
    t = (timesteps.float() / ntt).clamp(min=1e-8, max=1.0)
    snr = ((1.0 - t) ** 2) / (t ** 2 + 1e-8)
    capped = (snr > gamma).float()
    return float(capped.mean().item())


def trainer_applies_min_snr(gamma: float) -> bool:
    """Matches BaseSDTrainProcess / SDTrainer: SNR only when gamma > 1e-6."""
    return gamma is not None and gamma > 0.000001


def analyze_combo(
    use_dynamic_shifting: bool, gamma: float, h: int, w: int, pixel: int
) -> ScheduleStats:
    sched = make_scheduler(use_dynamic_shifting)
    timesteps = build_shift_schedule(sched, h, w)
    loss = torch.ones(timesteps.shape[0])
    snr = lookup_snr(None, sched, timesteps.tolist(), "cpu")

    if not trainer_applies_min_snr(gamma):
        # gamma=0 / null: toolkit skips apply_snr_weight — uniform MSE on schedule
        weighted = loss.clone()
        expected = loss.clone()
        formula_ok = True
        spot_ok = True
    else:
        weighted = apply_snr_weight(
            loss,
            timesteps,
            sched,
            gamma,
            prediction_type=flowmatch,
        )
        expected = expected_flow_match_min_snr_weight(snr, gamma)
        formula_ok = torch.allclose(weighted, expected, rtol=1e-4, atol=1e-6)

        # Spot-check arbitrary values on this schedule (interpolation path)
        try:
            assert_apply_snr_flow_match_weights(
                sched,
                SAMPLE_TS,
                gamma=gamma,
                device="cpu",
                verbose=False,
            )
            spot_ok = True
        except AssertionError:
            spot_ok = False

    return ScheduleStats(
        use_dynamic_shifting=use_dynamic_shifting,
        gamma=gamma,
        h=h,
        w=w,
        pixel=pixel,
        n_timesteps=int(timesteps.numel()),
        t_min=float(timesteps.min().item()),
        t_max=float(timesteps.max().item()),
        t_mean=float(timesteps.mean().item()),
        frac_capped_on_schedule=cap_fraction_on_schedule(timesteps, gamma),
        weight_min=float(weighted.min().item()),
        weight_max=float(weighted.max().item()),
        has_nan=bool(torch.isnan(weighted).any().item()),
        has_inf=bool(torch.isinf(weighted).any().item()),
        formula_ok=bool(formula_ok.item() if hasattr(formula_ok, "item") else formula_ok) and spot_ok,
    )


def print_table(rows: list[ScheduleStats]) -> None:
    hdr = (
        f"{'dynamic':>7} {'gamma':>5} {'res':>9} {'t_mean':>8} "
        f"{'cap%':>6} {'w_min':>8} {'w_max':>8} {'ok':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        res = f"{r.pixel}x{r.pixel}"
        cap_str = "   n/a" if r.gamma <= 0.000001 else f"{100 * r.frac_capped_on_schedule:6.1f}"
        print(
            f"{str(r.use_dynamic_shifting):>7} {r.gamma:5.1f} {res:>9} {r.t_mean:8.2f} "
            f"{cap_str} {r.weight_min:8.5f} {r.weight_max:8.5f} "
            f"{'yes' if r.formula_ok and not r.has_nan and not r.has_inf else 'NO':>4}"
        )


def main() -> None:
    rows: list[ScheduleStats] = []
    failures: list[str] = []

    for dynamic in (False, True):
        for gamma in GAMMAS:
            for h, w, pixel in RESOLUTIONS:
                stats = analyze_combo(dynamic, gamma, h, w, pixel)
                rows.append(stats)
                if stats.has_nan or stats.has_inf or not stats.formula_ok:
                    failures.append(
                        f"dynamic={dynamic} gamma={gamma} {pixel}x{pixel}: "
                        f"nan={stats.has_nan} inf={stats.has_inf} formula_ok={stats.formula_ok}"
                    )

    print("=== use_dynamic_shifting x min_snr_gamma compatibility ===\n")
    print_table(rows)

    # Cross-resolution stability for dynamic shifting
    dyn_by_gamma: dict[float, list[ScheduleStats]] = {g: [] for g in GAMMAS}
    static_by_gamma: dict[float, list[ScheduleStats]] = {g: [] for g in GAMMAS}
    for r in rows:
        (dyn_by_gamma if r.use_dynamic_shifting else static_by_gamma)[r.gamma].append(r)

    print("\n=== cap% spread across resolutions (higher = more timesteps hit min-SNR cap) ===")
    for gamma in GAMMAS:
        static_caps = [r.frac_capped_on_schedule for r in static_by_gamma[gamma]]
        dyn_caps = [r.frac_capped_on_schedule for r in dyn_by_gamma[gamma]]
        print(
            f"gamma={gamma:g}: static cap% {min(static_caps)*100:.1f}-{max(static_caps)*100:.1f} "
            f"(same for all res); dynamic cap% {min(dyn_caps)*100:.1f}-{max(dyn_caps)*100:.1f}"
        )

    print("\n=== dynamic shifting: mu and schedule delta 512 vs 1024 ===")
    from toolkit.samplers.custom_flowmatch_sampler import calculate_shift, flowmatch_image_seq_len

    for pixel, lh, lw in ((512, 64, 64), (1024, 128, 128)):
        isl = flowmatch_image_seq_len(lh, lw)
        mu = calculate_shift(isl, 256, 4096, 0.5, 1.15)
        print(f"  {pixel}x{pixel} (latent {lh}x{lw}): image_seq_len={isl}, mu={mu:.4f}")

    print("\n=== min_snr cap direction (low-noise vs high-noise) ===")
    sched = make_scheduler(True)
    build_shift_schedule(sched, 128, 128)  # 1024px
    gamma = 5.0
    low_t, high_t = 50.0, 950.0
    loss2 = torch.ones(2)
    w = apply_snr_weight(
        loss2,
        torch.tensor([low_t, high_t]),
        sched,
        gamma,
        prediction_type=flowmatch,
    )
    print(f"  t=50 (low noise):  weight={w[0].item():.6f} (should be capped)")
    print(f"  t=950 (high noise): weight={w[1].item():.6f} (should be 1.0)")

    print("\n=== gamma=0 note (trainer behaviour) ===")
    print("  min_snr_gamma=0: apply_snr_weight is NOT called (uniform loss on all timesteps).")
    w0 = apply_snr_weight(
        torch.ones(1),
        torch.tensor([500.0]),
        make_scheduler(True),
        0.0,
        prediction_type=flowmatch,
    )
    print(f"  If mis-applied with gamma=0, weight would be {w0.item():.6f} (zeros loss).")

    print("\n=== VERDICT ===")
    if failures:
        print("INCOMPATIBLE combinations detected:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print("All combinations are mechanically compatible (no NaN/Inf, apply_snr_weight matches formula).")
    print("\nRecommendations:")
    print("  - use_dynamic_shifting=false + min_snr_gamma: classic Flux toolkit path; gamma 5 is repo default.")
    print("  - use_dynamic_shifting=true + min_snr_gamma: valid; cap% grows with resolution and gamma.")
    print("  - gamma=0: no SNR reweighting (plain MSE); works with both shifting modes.")
    print("  - gamma=1: strongest downweight of clean timesteps; use if highlights blow out.")
    print("  - gamma=3: middle ground when gamma=5 feels too weak or gamma=1 too flat.")
    print("  - gamma=5: safest default with dynamic shifting (fewer capped steps at 1024).")
    print("  - Do not combine with use_diffsynth_training_loop=true (min_snr_gamma is disabled there).")
    print("  - Requires train.timestep_type: shift|flux_shift when use_dynamic_shifting=true.")


if __name__ == "__main__":
    main()
