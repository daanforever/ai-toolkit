"""
SNR / min_snr_gamma tests for Z-Image DiffSynth toolkit training loop.

When model.model_kwargs.use_diffsynth_training_loop is False, the model uses
CustomFlowMatchEulerDiscreteScheduler and ZImageDiffSynthTrainer leaves
min_snr_gamma / snr_gamma from train config enabled. SDTrainer.calculate_loss
then calls apply_snr_weight with train_config.prediction_type (e.g. flowmatch2).

Run from repo root with venv:
  venv\\Scripts\\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_snr_weighting.py -q
"""

import pytest
import torch

from extensions_built_in.diffusion_models.z_image_diffsynth.model import ZImageDiffSynthModel
from extensions_built_in.diffusion_models.z_image_diffsynth.scheduler_adapter import (
    DiffSynthZImageSchedulerAdapter,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.snr_weighting_checks import (
    assert_all_snr_table,
    assert_apply_snr_flow_match_weights,
    assert_scheduler_uses_compute_snr_path,
    expected_flow_match_min_snr_weight,
    lookup_snr,
    non_integer_schedule_timesteps,
)
from toolkit.train_tools import apply_snr_weight, get_all_snr
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


@pytest.fixture
def toolkit_scheduler():
    sched = ZImageDiffSynthModel.get_train_scheduler(use_diffsynth_loop=False)
    sched.set_train_timesteps(1000, device="cpu", timestep_type="linear")
    return sched


def test_get_train_scheduler_false_is_custom_flowmatch():
    sched = ZImageDiffSynthModel.get_train_scheduler(use_diffsynth_loop=False)
    assert isinstance(sched, CustomFlowMatchEulerDiscreteScheduler)
    assert not isinstance(sched, DiffSynthZImageSchedulerAdapter)


def test_compute_snr_table(toolkit_scheduler):
    assert_scheduler_uses_compute_snr_path(toolkit_scheduler)
    assert_all_snr_table(toolkit_scheduler, "cpu")


def test_apply_snr_weight_integer_timesteps(toolkit_scheduler):
    # assert_apply_snr_flow_match_weights compares each weighted loss to min-SNR expected values.
    assert_apply_snr_flow_match_weights(
        toolkit_scheduler,
        [10, 500, 990],
        gamma=5.0,
        device="cpu",
    )


def test_apply_snr_weight_float_timesteps_sigmoid(toolkit_scheduler):
    toolkit_scheduler.set_train_timesteps(32, device="cpu", timestep_type="sigmoid")
    float_timesteps = non_integer_schedule_timesteps(toolkit_scheduler, max_count=3)
    assert float_timesteps, "sigmoid schedule should yield non-integer timesteps"
    assert_apply_snr_flow_match_weights(
        toolkit_scheduler,
        float_timesteps,
        gamma=5.0,
        device="cpu",
    )


def test_apply_snr_weight_boundary_timesteps(toolkit_scheduler):
    assert_apply_snr_flow_match_weights(
        toolkit_scheduler,
        [1, 1000],
        gamma=5.0,
        device="cpu",
    )


def test_min_snr_caps_low_noise_not_high_noise(toolkit_scheduler):
    """min_snr_gamma must cap high-SNR (low timestep value), not high-noise values."""
    gamma = 5.0
    device = "cpu"
    low_noise_ts = 50.0
    high_noise_ts = 950.0
    all_snr = get_all_snr(toolkit_scheduler, device)
    snr_low = lookup_snr(all_snr, toolkit_scheduler, [low_noise_ts], device)[0]
    snr_high = lookup_snr(all_snr, toolkit_scheduler, [high_noise_ts], device)[0]
    assert snr_low.item() > gamma, "low-noise timestep should have SNR above gamma"
    assert snr_high.item() < gamma, "high-noise timestep should have SNR below gamma"

    loss = torch.ones(2, device=device)
    timesteps = torch.tensor([low_noise_ts, high_noise_ts], device=device)
    weighted = apply_snr_weight(
        loss,
        timesteps,
        toolkit_scheduler,
        gamma,
        prediction_type="flowmatch2",
    )
    expected_low = expected_flow_match_min_snr_weight(snr_low, gamma).item()
    expected_high = expected_flow_match_min_snr_weight(snr_high, gamma).item()
    capped_low = gamma / snr_low.item()

    assert weighted[0].item() == pytest.approx(expected_low, abs=1e-5)
    assert weighted[1].item() == pytest.approx(expected_high, abs=1e-5)
    assert weighted[1].item() == pytest.approx(1.0)
    assert weighted[0].item() < 1.0
    assert abs(weighted[0].item() - capped_low) < 1e-5, "ts=50 must use min-SNR weight gamma/snr"

    for g in (1.0, 5.0):
        w = apply_snr_weight(
            torch.ones(1, device=device),
            torch.tensor([high_noise_ts], device=device),
            toolkit_scheduler,
            g,
            prediction_type="flowmatch2",
        )
        assert w.item() == pytest.approx(1.0), f"ts=950 weight must be 1.0 for gamma={g}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
