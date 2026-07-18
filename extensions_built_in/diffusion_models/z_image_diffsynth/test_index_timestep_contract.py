"""
Regression: FlowMatch training must pass scheduler *values* (not slot indices)
to add_noise / predict / SNR.

Config under audit (toolkit loop):
  use_diffsynth_training_loop=false, use_dynamic_shifting=true,
  timestep_type=shift, content_or_style=balanced, noise_scheduler=flowmatch

Convention: slot 0 = high noise (~1000), slot ntt-1 = low noise (~1–3 on shift).
Confusing slot index 999 with timestep value ~999 is a common false alarm.

Related failure modes outside this path (not covered here):
  - LearnableSNRGamma: all_snr[t] when t is a scheduler value
  - BatchProcessor refiner double-up: randint indices → add_noise
  - Slider trainers: slot-like ints → add_noise without timesteps[i]

Run from repo root:
  venv\\Scripts\\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_index_timestep_contract.py -q
"""

from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models.z_image_diffsynth.model import ZImageDiffSynthModel
from toolkit.timestep_sampler import TimestepSampler


def _shift_scheduler_64x64():
    sched = ZImageDiffSynthModel.get_train_scheduler(
        use_diffsynth_loop=False,
        use_dynamic_shifting=True,
    )
    latents = torch.zeros(1, 16, 64, 64)
    sched.set_train_timesteps(
        1000,
        device="cpu",
        timestep_type="shift",
        latents=latents,
        patch_size=2,
    )
    return sched, latents


def test_balanced_sampler_returns_scheduler_values_not_slot_indices():
    """TimestepSampler must map slot indices → scheduler.timesteps[i] before return."""
    torch.manual_seed(0)
    sched, latents = _shift_scheduler_64x64()
    ntt = 1000
    train_config = SimpleNamespace(
        noise_scheduler="flowmatch",
        timestep_type="shift",
        num_train_timesteps=ntt,
        min_denoising_steps=0,
        max_denoising_steps=999,
    )
    sampler = TimestepSampler(train_config, sched)

    # Slot endpoints on the live schedule.
    assert float(sched.timesteps[0].item()) > 900.0
    assert float(sched.timesteps[ntt - 1].item()) < 50.0

    result = sampler.sample(
        batch_size=64,
        latents=latents,
        content_or_style="balanced",
        min_noise_steps=0,
        max_noise_steps=999,
        num_train_timesteps=ntt,
        device=torch.device("cpu"),
        step_num=0,
    )
    timesteps = result.timesteps
    indices = result.timestep_indices
    assert indices is not None

    # Returned tensor must be schedule *values*, not raw arange of slots.
    assert not torch.equal(
        timesteps.cpu().float(),
        indices.cpu().float(),
    ), "timesteps must not equal slot indices (would invert high/low noise in add_noise)"

    # Each sampled value must match the schedule at that slot.
    expected = sched.timesteps[indices.long()].cpu()
    assert torch.allclose(timesteps.cpu().float(), expected.float(), atol=1e-5)

    # Force high-noise and low-noise slots through the same mapping used in sample().
    t_hi = float(sched.timesteps[0].item())
    t_lo = float(sched.timesteps[ntt - 1].item())
    assert t_hi > 900.0
    assert t_lo < 50.0


def test_add_noise_uses_timestep_value_not_slot_index():
    """CustomFlowMatch.add_noise: value~1000 → pure noise; value~low → near clean."""
    sched, _ = _shift_scheduler_64x64()
    x0 = torch.zeros(1, 4, 8, 8)
    noise = torch.ones(1, 4, 8, 8)

    t_hi = sched.timesteps[0:1]
    t_lo = sched.timesteps[-1:]
    noisy_hi = sched.add_noise(x0, noise, t_hi)
    noisy_lo = sched.add_noise(x0, noise, t_lo)

    assert torch.allclose(noisy_hi, noise, atol=0.05), (
        f"high-noise value {float(t_hi.item()):.1f} should yield near-noise latents"
    )
    assert torch.allclose(noisy_lo, x0, atol=0.05), (
        f"low-noise value {float(t_lo.item()):.1f} should yield near-clean latents"
    )


def test_passing_slot_index_as_timestep_value_inverts_noise():
    """
    Documents the failure class: if slot index 5 is passed where a scheduler
    value (~995) is required, add_noise treats t=5/1000 ≈ clean instead of noisy.
    """
    sched, _ = _shift_scheduler_64x64()
    x0 = torch.zeros(1, 4, 8, 8)
    noise = torch.ones(1, 4, 8, 8)

    slot_index = 5
    value_at_slot = float(sched.timesteps[slot_index].item())
    assert value_at_slot > 900.0, "slot 5 must be high-noise on descending shift schedule"

    # Correct: use scheduler value at the slot.
    correct = sched.add_noise(
        x0, noise, torch.tensor([value_at_slot], dtype=torch.float32)
    )
    # Bug: pass raw slot index as if it were a timestep value.
    buggy = sched.add_noise(
        x0, noise, torch.tensor([float(slot_index)], dtype=torch.float32)
    )

    assert torch.allclose(correct, noise, atol=0.05)
    assert torch.allclose(buggy, x0, atol=0.05), (
        "passing slot index as value must produce near-clean (inverted) noise mix"
    )
    assert not torch.allclose(correct, buggy, atol=0.1)
