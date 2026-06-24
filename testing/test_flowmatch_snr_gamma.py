"""
Flow-match min_snr_gamma regression for z_image_diffsynth toolkit-loop configs.

Reproduces the user report: prediction_type=flowmatch, timestep_type=shift,
use_dynamic_shifting=true, min_snr_gamma changes should NOT affect high-noise
scheduler *values* (~990–1000), but *do* affect low-noise values and slot 999
(which maps to timestep ~3, not ~999).

Run from repo root:
  venv\\Scripts\\python.exe -m pytest testing/test_flowmatch_snr_gamma.py -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from extensions_built_in.diffusion_models.z_image_diffsynth.scheduler_config import build_scheduler_config
from extensions_built_in.diffusion_models.z_image_diffsynth.snr_weighting_checks import (
    assert_apply_snr_flow_match_weights,
    expected_flowmatch_inverted_u_weight,
    expected_flowmatch_snr_weight,
    lookup_snr,
)
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.train_tools import apply_snr_weight


def _user_shift_scheduler(resolution_px: int = 1024) -> CustomFlowMatchEulerDiscreteScheduler:
    sched = CustomFlowMatchEulerDiscreteScheduler(**build_scheduler_config(True))
    latent = resolution_px // 8
    latents = torch.zeros(1, 16, latent, latent)
    sched.set_train_timesteps(1000, device="cpu", timestep_type="shift", latents=latents)
    return sched


def _weights_for_gammas(
    scheduler,
    timestep_values: list[float],
    gammas: list[float],
    *,
    prediction_type: str = "flowmatch",
) -> dict[float, torch.Tensor]:
    ts = torch.tensor(timestep_values, dtype=torch.float32)
    loss = torch.ones(len(timestep_values), dtype=torch.float32)
    return {
        g: apply_snr_weight(loss, ts, scheduler, g, prediction_type=prediction_type)
        for g in gammas
    }


@pytest.fixture
def shift_scheduler_1024():
    return _user_shift_scheduler(1024)


def test_flowmatch_formula_matches_apply_snr_weight(shift_scheduler_1024):
    assert_apply_snr_flow_match_weights(
        shift_scheduler_1024,
        [990.0, 966.0, 441.0, 3.15],
        gamma=1.0,
        device="cpu",
        prediction_type="flowmatch",
    )


def test_high_noise_timestep_values_are_gamma_independent(shift_scheduler_1024):
    """Scheduler scalars ~990–1000: min(gamma, snr)=snr, so gamma must not matter."""
    gammas = [1.0, 5.0]
    high_noise_ts = [999.0, 990.0, 966.0, 900.0, 800.0]
    by_gamma = _weights_for_gammas(shift_scheduler_1024, high_noise_ts, gammas)
    for ts in high_noise_ts:
        i = high_noise_ts.index(ts)
        w1 = by_gamma[1.0][i].item()
        w5 = by_gamma[5.0][i].item()
        assert w1 == pytest.approx(w5, rel=0, abs=1e-12), (
            f"timestep value {ts}: flowmatch weight must not depend on gamma "
            f"(w@gamma=1: {w1}, w@gamma=5: {w5})"
        )


def test_slot_999_is_low_noise_and_gamma_dependent(shift_scheduler_1024):
    """
    On shift schedules slot 0 = max noise (t≈1000), slot 999 = min noise (t≈3).
    Confusing slot index 999 with timestep value ~999 causes false bug reports.
    """
    sched = shift_scheduler_1024
    slot = 999
    ts = float(sched.timesteps[slot].item())
    assert ts < 10.0, f"slot 999 must be low-noise end, got timestep value {ts}"
    snr = lookup_snr(None, sched, [ts], "cpu")[0].item()
    assert snr > 1.0

    w1 = apply_snr_weight(
        torch.ones(1), torch.tensor([ts]), sched, 1.0, prediction_type="flowmatch"
    )[0].item()
    w5 = apply_snr_weight(
        torch.ones(1), torch.tensor([ts]), sched, 5.0, prediction_type="flowmatch"
    )[0].item()
    assert w5 > w1 > 0.0
    assert w5 == pytest.approx(5.0 * w1, rel=1e-5)


def test_flowmatch_vs_flowmatch2_differ_at_high_noise(shift_scheduler_1024):
    """flowmatch down-weights high noise; flowmatch2 keeps weight 1.0 there."""
    ts = 966.0
    snr = lookup_snr(None, shift_scheduler_1024, [ts], "cpu")
    w_fm = apply_snr_weight(
        torch.ones(1),
        torch.tensor([ts]),
        shift_scheduler_1024,
        1.0,
        prediction_type="flowmatch",
    )[0].item()
    w_fm2 = apply_snr_weight(
        torch.ones(1),
        torch.tensor([ts]),
        shift_scheduler_1024,
        1.0,
        prediction_type="flowmatch2",
    )[0].item()
    assert w_fm < 0.01
    assert w_fm2 == pytest.approx(1.0)
    assert w_fm == pytest.approx(
        expected_flowmatch_inverted_u_weight(snr, 1.0).item(), rel=1e-5
    )
    assert w_fm2 == pytest.approx(
        expected_flowmatch_snr_weight(snr, 1.0, prediction_type="flowmatch2").item(),
        rel=1e-5,
    )


def test_gamma_cap_region_starts_below_timestep_800_for_gamma_1(shift_scheduler_1024):
    """min(gamma, snr) uses gamma when snr > gamma (low noise / small t values)."""
    sched = shift_scheduler_1024
    ts_low_noise = 441.0
    snr = lookup_snr(None, sched, [ts_low_noise], "cpu")[0].item()
    assert snr > 1.0
    w1 = apply_snr_weight(
        torch.ones(1), torch.tensor([ts_low_noise]), sched, 1.0, prediction_type="flowmatch"
    )[0].item()
    w5 = apply_snr_weight(
        torch.ones(1), torch.tensor([ts_low_noise]), sched, 5.0, prediction_type="flowmatch"
    )[0].item()
    assert w1 != pytest.approx(w5, rel=1e-6)
