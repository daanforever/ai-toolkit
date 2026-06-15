"""
SNR / min_snr_gamma tests for Z-Image DiffSynth toolkit training loop.

When model.model_kwargs.use_diffsynth_training_loop is False, the model uses
CustomFlowMatchEulerDiscreteScheduler and ZImageDiffSynthTrainer leaves
min_snr_gamma / snr_gamma from train config enabled. SDTrainer.calculate_loss
then calls apply_snr_weight with prediction_type="flow_match".

Run from repo root with venv:
  venv\\Scripts\\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_snr_weighting.py -q
"""

import pytest

from extensions_built_in.diffusion_models.z_image_diffsynth.model import ZImageDiffSynthModel
from extensions_built_in.diffusion_models.z_image_diffsynth.scheduler_adapter import (
    DiffSynthZImageSchedulerAdapter,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.snr_weighting_checks import (
    assert_all_snr_table,
    assert_apply_snr_flow_match_weights,
    assert_scheduler_uses_compute_snr_path,
    non_integer_schedule_timesteps,
)
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


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
