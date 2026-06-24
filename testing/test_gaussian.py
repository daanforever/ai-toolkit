"""Unit tests for gaussian timestep weighting/sampling."""

import os
import sys

import pytest
import torch

# Add project root to sys.path for `import toolkit...` when running tests from repo root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.timestep_sampler import TimestepSampler, allowed_slot_index_range
from extensions_built_in.sd_trainer.gaussian_timestep_weights import (
    evaluate_gaussian_timestep,
    evaluate_gaussian_timestep_bimodal,
    scheduler_timesteps_align_with_index_grid,
    timestep_values_to_slot_indices,
)


class _DummyTrainConfig:
    # Ensure we hit the gaussian branch inside `TimestepSampler.sample()`.
    timestep_type = "shift"

    num_train_timesteps = 1000
    gaussian_mean = 450.0
    gaussian_std = 0.45
    gaussian_mean_2 = 750.0
    gaussian_std_2 = 0.35


class _DummyNoiseScheduler:
    def __init__(self, timesteps: torch.Tensor):
        self.timesteps = timesteps


def test_evaluate_gaussian_timestep_peak_near_mean():
    ntt = 1000
    mu = 450.0
    sigma = 0.45

    timesteps = torch.arange(ntt, dtype=torch.float32)
    weights = evaluate_gaussian_timestep(
        timesteps=timesteps,
        mu=mu,
        sigma=sigma,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_train_timesteps=ntt,
    )
    probs = weights / weights.sum().clamp(min=1e-8)

    assert float(weights.min().item()) >= 0.0
    assert float(weights.max().item()) <= 1.0

    # Peak should be very close to `mu` (discrete grid).
    argmax_t = int(torch.argmax(probs).item())
    assert abs(argmax_t - int(mu)) <= 1

    # Probabilities should be normalized.
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)


def test_timestep_sampler_gaussian_matches_expected_mean():
    torch.manual_seed(0)

    ntt = 1000
    mu = 450.0
    sigma = 0.45

    # Choose a realistic denoising range so indices stay within scheduler bounds.
    min_noise_steps = 1
    max_noise_steps = 999

    # Fake scheduler where timestep values equal their indices: timesteps[i] = i.
    noise_scheduler = _DummyNoiseScheduler(torch.arange(ntt, dtype=torch.float32))

    train_config = _DummyTrainConfig()
    train_config.gaussian_mean = mu
    train_config.gaussian_std = sigma
    train_config.num_train_timesteps = ntt

    sampler = TimestepSampler(train_config, noise_scheduler)

    batch_size = 30000
    latents = torch.empty((batch_size, 1), device=torch.device("cpu"))

    # `timesteps` is sampled with replacement; mean should be close to the theoretical mean.
    result = sampler.sample(
        batch_size=batch_size,
        latents=latents,
        content_or_style="gaussian",
        min_noise_steps=min_noise_steps,
        max_noise_steps=max_noise_steps,
        num_train_timesteps=ntt,
        device=torch.device("cpu"),
        step_num=0,
    )
    sampled_timesteps = result.timesteps
    sampled_mean = sampled_timesteps.mean().item()

    allowed_start = ntt - max_noise_steps
    allowed_end = ntt - min_noise_steps
    allowed_indices = torch.arange(allowed_start, allowed_end + 1, dtype=torch.long)
    allowed_timestep_values = noise_scheduler.timesteps[allowed_indices]

    weights = evaluate_gaussian_timestep(
        timesteps=allowed_timestep_values,
        mu=mu,
        sigma=sigma,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_train_timesteps=ntt,
    )
    probs = weights / weights.sum().clamp(min=1e-8)
    expected_mean = (allowed_timestep_values * probs).sum().item()

    # Monte-Carlo tolerance: should be tight enough to catch regressions, loose enough to be stable.
    assert abs(sampled_mean - expected_mean) < 2.0


def test_evaluate_gaussian_timestep_bimodal_two_peaks():
    ntt = 1000
    mu1, mu2 = 200.0, 800.0
    s1, s2 = 0.12, 0.15
    timesteps = torch.arange(ntt, dtype=torch.float32)
    weights = evaluate_gaussian_timestep_bimodal(
        timesteps=timesteps,
        mu1=mu1,
        sigma1=s1,
        mu2=mu2,
        sigma2=s2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_train_timesteps=ntt,
    )
    # Global max-normalization can make one shoulder outrank the second peak in raw top-k;
    # check argmax in windows around each configured mean.
    lo1, hi1 = max(0, int(mu1) - 80), min(ntt, int(mu1) + 80)
    lo2, hi2 = max(0, int(mu2) - 80), min(ntt, int(mu2) + 80)
    peak1_idx = lo1 + int(torch.argmax(weights[lo1:hi1]).item())
    peak2_idx = lo2 + int(torch.argmax(weights[lo2:hi2]).item())
    assert float(weights.min().item()) >= 0.0
    assert float(weights.max().item()) <= 1.0
    assert abs(peak1_idx - mu1) <= 25
    assert abs(peak2_idx - mu2) <= 25


def test_timestep_sampler_gaussian_bimodal_expected_mean():
    torch.manual_seed(1)
    ntt = 1000
    mu1, mu2 = 250.0, 750.0
    s1, s2 = 0.15, 0.18
    min_noise_steps = 1
    max_noise_steps = 999
    noise_scheduler = _DummyNoiseScheduler(torch.arange(ntt, dtype=torch.float32))
    train_config = _DummyTrainConfig()
    train_config.gaussian_mean = mu1
    train_config.gaussian_std = s1
    train_config.gaussian_mean_2 = mu2
    train_config.gaussian_std_2 = s2
    train_config.num_train_timesteps = ntt
    sampler = TimestepSampler(train_config, noise_scheduler)
    batch_size = 40000
    latents = torch.empty((batch_size, 1), device=torch.device("cpu"))
    result = sampler.sample(
        batch_size=batch_size,
        latents=latents,
        content_or_style="gaussian_bimodal",
        min_noise_steps=min_noise_steps,
        max_noise_steps=max_noise_steps,
        num_train_timesteps=ntt,
        device=torch.device("cpu"),
        step_num=0,
    )
    allowed_start = ntt - max_noise_steps
    allowed_end = ntt - min_noise_steps
    allowed_indices = torch.arange(allowed_start, allowed_end + 1, dtype=torch.long)
    allowed_timestep_values = noise_scheduler.timesteps[allowed_indices]
    w = evaluate_gaussian_timestep_bimodal(
        allowed_timestep_values,
        mu1,
        s1,
        mu2,
        s2,
        torch.device("cpu"),
        torch.float32,
        ntt,
    )
    probs = w / w.sum().clamp(min=1e-8)
    expected_mean = (allowed_timestep_values * probs).sum().item()
    assert abs(result.timesteps.mean().item() - expected_mean) < 3.0


def test_bimodal_identical_peaks_matches_unimodal_weights():
    """50/50 mixture of the same component equals that component (weights on grid)."""
    ntt = 1000
    mu, sigma = 412.0, 0.33
    t = torch.arange(ntt, dtype=torch.float32)
    w_uni = evaluate_gaussian_timestep(
        t, mu, sigma, torch.device("cpu"), torch.float32, ntt
    )
    w_bi = evaluate_gaussian_timestep_bimodal(
        t, mu, sigma, mu, sigma, torch.device("cpu"), torch.float32, ntt
    )
    assert torch.allclose(w_uni, w_bi, rtol=0, atol=1e-6)


def test_evaluate_gaussian_clamps_timestep_values_to_grid():
    """Scheduler values outside 0..ntt-1 must not crash; lookup clamps to grid ends."""
    ntt = 500
    mu, sigma = 200.0, 0.2
    weird = torch.tensor([-50.0, 0.0, 250.0, 999.0, 5000.0], dtype=torch.float32)
    w = evaluate_gaussian_timestep(
        weird, mu, sigma, torch.device("cpu"), torch.float32, ntt
    )
    t_grid = torch.arange(ntt, dtype=torch.float32)
    w_full = evaluate_gaussian_timestep(
        t_grid, mu, sigma, torch.device("cpu"), torch.float32, ntt
    )
    assert torch.isclose(w[0], w_full[0])
    assert torch.isclose(w[1], w_full[0])
    assert torch.isclose(w[2], w_full[250])
    assert torch.isclose(w[3], w_full[ntt - 1])
    assert torch.isclose(w[4], w_full[ntt - 1])


def test_allowed_slot_index_range_min_denoising_zero_clamps_hi():
    assert allowed_slot_index_range(1000, 0, 999) == (1, 999)
    assert allowed_slot_index_range(1000, 1, 999) == (1, 999)


def test_balanced_flowmatch_range_keeps_upper_bound_inclusive():
    torch.manual_seed(0)
    ntt = 1000
    schedule = torch.linspace(1000, 1, ntt, dtype=torch.float32)
    noise_scheduler = _DummyNoiseScheduler(schedule)
    train_config = _DummyTrainConfig()
    train_config.noise_scheduler = "flowmatch"
    sampler = TimestepSampler(train_config, noise_scheduler)

    sampled_indices = sampler._sample_balanced(
        batch_size=256,
        min_noise_steps=1,
        max_noise_steps=2,
        device=torch.device("cpu"),
    )
    assert int(sampled_indices.min().item()) == 0
    assert int(sampled_indices.max().item()) == 1


def test_balanced_flowmatch_equal_min_max_maps_to_first_slot():
    ntt = 1000
    schedule = torch.linspace(1000, 1, ntt, dtype=torch.float32)
    noise_scheduler = _DummyNoiseScheduler(schedule)
    train_config = _DummyTrainConfig()
    train_config.noise_scheduler = "flowmatch"
    sampler = TimestepSampler(train_config, noise_scheduler)

    sampled_indices = sampler._sample_balanced(
        batch_size=16,
        min_noise_steps=1,
        max_noise_steps=1,
        device=torch.device("cpu"),
    )
    assert torch.all(sampled_indices == 0)


def test_gaussian_bimodal_min_denoising_steps_zero_no_oob():
    """min_denoising_steps=0 used to extend arange to index ntt; must not crash or OOB."""
    torch.manual_seed(0)
    ntt = 1000
    sched = torch.linspace(1000, 1, ntt, dtype=torch.float32)
    noise_scheduler = _DummyNoiseScheduler(sched)
    train_config = _DummyTrainConfig()
    train_config.gaussian_mean = 300.0
    train_config.gaussian_std = 0.2
    train_config.gaussian_mean_2 = 800.0
    train_config.gaussian_std_2 = 0.2
    train_config.num_train_timesteps = ntt
    sampler = TimestepSampler(train_config, noise_scheduler)
    result = sampler.sample(
        batch_size=128,
        latents=torch.empty((128, 1), device=torch.device("cpu")),
        content_or_style="gaussian_bimodal",
        min_noise_steps=0,
        max_noise_steps=999,
        num_train_timesteps=ntt,
        device=torch.device("cpu"),
        step_num=0,
    )
    assert torch.isfinite(result.timesteps).all()
    assert (result.timesteps >= sched[-1]).all() and (result.timesteps <= sched[0]).all()


def test_timestep_sampler_gaussian_bimodal_narrow_window_stays_in_bounds():
    """When the allowed index range is tiny, sampling still yields valid indices."""
    torch.manual_seed(2)
    ntt = 1000
    noise_scheduler = _DummyNoiseScheduler(torch.arange(ntt, dtype=torch.float32))
    train_config = _DummyTrainConfig()
    train_config.gaussian_mean = 100.0
    train_config.gaussian_std = 0.1
    train_config.gaussian_mean_2 = 900.0
    train_config.gaussian_std_2 = 0.1
    train_config.num_train_timesteps = ntt
    sampler = TimestepSampler(train_config, noise_scheduler)
    # Only indices 400..600 → both sharp peaks are outside; weights flat-ish but valid.
    result = sampler.sample(
        batch_size=500,
        latents=torch.empty((500, 1), device=torch.device("cpu")),
        content_or_style="gaussian_bimodal",
        min_noise_steps=400,
        max_noise_steps=600,
        num_train_timesteps=ntt,
        device=torch.device("cpu"),
        step_num=0,
    )
    lo, hi = ntt - 600, ntt - 400
    assert result.timesteps.min() >= noise_scheduler.timesteps[lo]
    assert result.timesteps.max() <= noise_scheduler.timesteps[hi]


def test_gaussian_bimodal_flowmatch_sigma_must_not_be_used_as_grid_index():
    """
    CustomFlowMatchEulerDiscreteScheduler uses timestep *values* ~1000→1. Gaussian weights are
    defined on discrete training slots 0..ntt-1 (same axis as gaussian_mean / gaussian_mean_2).

    Passing the float sigma (e.g. ~43.6) into evaluate_gaussian_timestep_bimodal() uses int(43)
    as a grid row — far from both peaks (300, 800) — and yields an inappropriately low weight.

    For the same batch row, timestep_index 987 should be looked up as slot 987 on that grid,
    not as value 43.6. Regression: observed (43.60465 → ~0.377) vs slot 987 → higher weight.
    """
    ntt = 1000
    mu1, s1, mu2, s2 = 300.0, 0.2, 800.0, 0.2
    device = torch.device("cpu")
    dtype = torch.float32

    # From user log: actual timestep tensor value after indexing noise_scheduler.timesteps.
    flow_sigma = torch.tensor([43.604652404785156], dtype=dtype)
    w_if_treat_sigma_as_index = evaluate_gaussian_timestep_bimodal(
        flow_sigma, mu1, s1, mu2, s2, device, dtype, ntt
    )
    # Same row: discrete scheduler index from the user's log.
    w_for_step_slot_987 = evaluate_gaussian_timestep_bimodal(
        torch.tensor([987.0], dtype=dtype), mu1, s1, mu2, s2, device, dtype, ntt
    )

    assert abs(w_if_treat_sigma_as_index.item() - 0.37732598185539246) < 1e-5
    assert w_for_step_slot_987.item() > w_if_treat_sigma_as_index.item() + 0.2
    assert w_for_step_slot_987.item() > 0.55

    # Flow-style schedule: value at index 987 is ~13, not 43.6 — still must not use value as slot.
    sched = torch.linspace(1000, 1, ntt, dtype=dtype)
    w_at_linspace_987 = evaluate_gaussian_timestep_bimodal(
        sched[987].unsqueeze(0), mu1, s1, mu2, s2, device, dtype, ntt
    )
    w_slot_987_again = evaluate_gaussian_timestep_bimodal(
        torch.tensor([987.0], dtype=dtype), mu1, s1, mu2, s2, device, dtype, ntt
    )
    assert w_at_linspace_987.item() < w_slot_987_again.item() - 0.1

    # Correct usage: map timestep values back to their slot indices using the actual scheduler `timesteps`.
    # We can't reconstruct the full user's scheduler here, but if `schedule[987] == flow_sigma`,
    # then mapping must return 987 and the resulting weight must match `w_for_step_slot_987`.
    custom_schedule = torch.linspace(1000, 1, ntt, dtype=dtype)
    custom_schedule[987] = flow_sigma.item()
    mapped_slot = timestep_values_to_slot_indices(
        flow_sigma, custom_schedule, ntt=ntt
    )
    assert int(mapped_slot.item()) == 987
    w_mapped = evaluate_gaussian_timestep_bimodal(
        mapped_slot, mu1, s1, mu2, s2, device, dtype, ntt
    )
    assert torch.isclose(w_mapped, w_for_step_slot_987, atol=1e-6)


def test_scheduler_timesteps_align_with_index_grid_true_for_arange():
    ntt = 1000
    schedule = torch.arange(ntt, dtype=torch.float32)
    assert scheduler_timesteps_align_with_index_grid(schedule, ntt) is True


def test_scheduler_timesteps_align_with_index_grid_false_for_linspace():
    ntt = 1000
    schedule = torch.linspace(1000, 1, ntt, dtype=torch.float32)
    assert scheduler_timesteps_align_with_index_grid(schedule, ntt) is False


def test_evaluate_gaussian_bimodal_flowmatch_resolves_means_as_scheduler_values():
    """With noise_scheduler_timesteps=linspace(1000→1), μ are values → peaks near matching slots."""
    ntt = 1000
    sched = torch.linspace(1000, 1, ntt, dtype=torch.float32)
    t = torch.arange(ntt, dtype=torch.float32)
    w = evaluate_gaussian_timestep_bimodal(
        t,
        300.0,
        0.05,
        850.0,
        0.05,
        torch.device("cpu"),
        torch.float32,
        ntt,
        noise_scheduler_timesteps=sched,
    )
    assert w[700] > w[300]
    assert w[150] > w[850]


def test_timestep_values_to_slot_indices_maps_back_schedule_values():
    ntt = 1000
    schedule = torch.linspace(1000, 1, ntt, dtype=torch.float32)
    slot = 987
    value = schedule[slot].unsqueeze(0)
    mapped = timestep_values_to_slot_indices(value, schedule, ntt=ntt)
    assert int(mapped.item()) == slot


def test_map_then_evaluate_matches_slot_weight():
    ntt = 1000
    mu1, s1, mu2, s2 = 300.0, 0.2, 800.0, 0.2
    device = torch.device("cpu")
    dtype = torch.float32

    schedule = torch.linspace(1000, 1, ntt, dtype=dtype)
    slot = 987
    value = schedule[slot].unsqueeze(0)
    mapped = timestep_values_to_slot_indices(value, schedule, ntt=ntt)

    w_slot = evaluate_gaussian_timestep_bimodal(
        torch.tensor([float(slot)]), mu1, s1, mu2, s2, device, dtype, ntt
    )
    w_mapped = evaluate_gaussian_timestep_bimodal(
        mapped, mu1, s1, mu2, s2, device, dtype, ntt
    )
    assert torch.isclose(w_mapped, w_slot, atol=1e-6)

def test_gaussian_bimodal_content_or_style_overrides_one_step_timestep_type():
    """If UI/runtime sets timestep_type=one_step, bimodal sampling must still follow content_or_style."""
    torch.manual_seed(0)
    ntt = 1000
    sched = torch.linspace(1000, 1, ntt, dtype=torch.float32)
    noise_scheduler = _DummyNoiseScheduler(sched)
    train_config = _DummyTrainConfig()
    train_config.timestep_type = "one_step"
    train_config.gaussian_mean = 300.0
    train_config.gaussian_std = 0.2
    train_config.gaussian_mean_2 = 800.0
    train_config.gaussian_std_2 = 0.2
    train_config.num_train_timesteps = ntt
    sampler = TimestepSampler(train_config, noise_scheduler)
    result = sampler.sample(
        batch_size=4000,
        latents=torch.empty((4000, 1)),
        content_or_style="gaussian_bimodal",
        min_noise_steps=5,
        max_noise_steps=995,
        num_train_timesteps=ntt,
        device=torch.device("cpu"),
        step_num=0,
    )
    # Pure one_step: every index 0 → timestep value 1000, zero variance.
    assert result.timesteps.float().std().item() > 40.0
    assert result.timesteps.min().item() < 450.0


def test_gaussian_timestep_type_invalid_for_scheduler():
    from extensions_built_in.diffusion_models.z_image_diffsynth.scheduler_config import (
        build_scheduler_config,
    )
    from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

    sched = CustomFlowMatchEulerDiscreteScheduler(**build_scheduler_config(False))
    with pytest.raises(ValueError, match="Invalid timestep type"):
        sched.set_train_timesteps(100, device="cpu", timestep_type="gaussian_bimodal")


def test_shift_schedule_independent_of_timestep_weighting_field():
    """timestep_weighting is not passed to set_train_timesteps; shift grid is unchanged."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.scheduler_config import (
        build_scheduler_config,
    )
    from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

    latents = torch.zeros(1, 16, 64, 64)
    sched = CustomFlowMatchEulerDiscreteScheduler(**build_scheduler_config(False))
    t_shift = sched.set_train_timesteps(
        1000, device="cpu", timestep_type="shift", latents=latents
    )
    sched2 = CustomFlowMatchEulerDiscreteScheduler(**build_scheduler_config(False))
    t_shift_again = sched2.set_train_timesteps(
        1000, device="cpu", timestep_type="shift", latents=latents
    )
    assert torch.equal(t_shift, t_shift_again)
    linear_sched = CustomFlowMatchEulerDiscreteScheduler(**build_scheduler_config(False))
    t_linear = linear_sched.set_train_timesteps(1000, device="cpu", timestep_type="linear")
    assert not torch.allclose(t_shift, t_linear)


def test_weighted_timestep_type_invalid_for_scheduler():
    from extensions_built_in.diffusion_models.z_image_diffsynth.scheduler_config import (
        build_scheduler_config,
    )
    from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

    sched = CustomFlowMatchEulerDiscreteScheduler(**build_scheduler_config(False))
    with pytest.raises(ValueError, match="Invalid timestep type"):
        sched.set_train_timesteps(100, device="cpu", timestep_type="weighted")


def test_shift_schedule_with_weighted_scheme_uses_default_weighing():
    from extensions_built_in.diffusion_models.z_image_diffsynth.scheduler_config import (
        build_scheduler_config,
    )
    from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
    from toolkit.timestep_weighing.default_weighing_scheme import default_weighing_scheme

    latents = torch.zeros(1, 16, 64, 64)
    sched = CustomFlowMatchEulerDiscreteScheduler(**build_scheduler_config(False))
    sched.set_train_timesteps(1000, device="cpu", timestep_type="shift", latents=latents)
    sample_ts = sched.timesteps[:3]
    weights = sched.get_weights_for_timesteps(sample_ts, timestep_type="weighted")
    expected = torch.tensor(
        [default_weighing_scheme[i] for i in range(3)],
        device=sample_ts.device,
        dtype=sample_ts.dtype,
    )
    assert torch.allclose(weights, expected)

