"""
Print flow-match min-SNR weights for scheduler timesteps every N training slots.

Run from repo root:
  venv\\Scripts\\python.exe extensions_built_in/diffusion_models/z_image_diffsynth/print_flowmatch_snr_weights.py
  venv\\Scripts\\python.exe extensions_built_in/diffusion_models/z_image_diffsynth/print_flowmatch_snr_weights.py --dynamic-shifting --resolution 1024 --gamma 5
  venv\\Scripts\\python.exe extensions_built_in/diffusion_models/z_image_diffsynth/print_flowmatch_snr_weights.py --matrix --dynamic-shifting --resolution 1024 --gammas 0.25,0.5,1,3,5,10 --prediction-type flowmatch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.diffusion_models.z_image_diffsynth.scheduler_config import build_scheduler_config
from extensions_built_in.diffusion_models.z_image_diffsynth.snr_weighting_checks import (
    print_flowmatch_snr_weight_matrix,
    print_flowmatch_snr_weight_table,
)
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


def _parse_gammas(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print flow-match SNR weights every N scheduler slots.")
    p.add_argument("--gamma", type=float, default=5.0, help="min_snr_gamma for single-gamma table (default: 5)")
    p.add_argument(
        "--gammas",
        type=_parse_gammas,
        default=None,
        help="comma-separated min_snr_gamma values for --matrix (e.g. 0.25,0.5,1,3,5,10)",
    )
    p.add_argument(
        "--matrix",
        action="store_true",
        help="CSV matrix: timestep, snr, <gamma columns> (requires --gammas)",
    )
    p.add_argument(
        "--prediction-type",
        default="flowmatch",
        help="apply_snr_weight prediction_type (default: flowmatch)",
    )
    p.add_argument("--step", type=int, default=100, help="slot stride (default: 100 → slots 0,100,…,999)")
    p.add_argument(
        "--timestep-type",
        choices=("shift", "linear", "sigmoid"),
        default="shift",
        help="CustomFlowMatch set_train_timesteps mode (default: shift)",
    )
    p.add_argument(
        "--dynamic-shifting",
        action="store_true",
        help="use_dynamic_shifting=true (requires shift timestep-type)",
    )
    p.add_argument(
        "--resolution",
        type=int,
        choices=(512, 1024),
        default=512,
        help="latent resolution for shift schedule (default: 512)",
    )
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    latent = args.resolution // 8
    sched = CustomFlowMatchEulerDiscreteScheduler(**build_scheduler_config(args.dynamic_shifting))
    latents = None
    if args.timestep_type == "shift":
        latents = torch.zeros(1, 16, latent, latent)
    sched.set_train_timesteps(
        1000,
        device=args.device,
        timestep_type=args.timestep_type,
        latents=latents,
    )
    t_min = sched.timesteps.min().item()
    t_max = sched.timesteps.max().item()
    shift_mode = "dynamic" if args.dynamic_shifting else "static"
    print(
        f"FlowMatch scheduler: prediction_type={args.prediction_type!r}, "
        f"timestep_type={args.timestep_type!r}, "
        f"shift={shift_mode}, resolution={args.resolution}px, "
        f"schedule t=[{t_min:.4g}, {t_max:.4g}], slots every {args.step}",
        file=sys.stderr,
    )
    if args.matrix:
        if not args.gammas:
            raise SystemExit("--matrix requires --gammas (comma-separated values)")
        print_flowmatch_snr_weight_matrix(
            sched,
            gammas=args.gammas,
            device=args.device,
            slot_step=args.step,
            prediction_type=args.prediction_type,
        )
    else:
        print_flowmatch_snr_weight_table(
            sched,
            gamma=args.gamma,
            device=args.device,
            slot_step=args.step,
            prediction_type=args.prediction_type,
        )


if __name__ == "__main__":
    main()
