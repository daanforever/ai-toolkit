"""
CPU unit tests for timestep_type=turbo_prior (Turbo 8-NFE grid + Voronoi jitter).

No model weights. Style mirrors test_index_timestep_contract.py.

Run from repo root:
  venv\\Scripts\\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_turbo_prior.py -q
"""

from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.diffusion_models.z_image_diffsynth.turbo_schedule import (
    TURBO_BALANCED_TARGET_T,
    get_turbo_sigmas_and_timesteps,
    turbo_balanced_target_slot,
    turbo_slot_dsigma_weights,
    turbo_slot_sampling_weights,
)
from toolkit.timestep_sampler import TimestepSampler

# Official static-shift 8-NFE Turbo centers (atol=1).
EXPECTED_CENTERS = [1000, 955, 900, 833, 750, 643, 500, 300]


def _centers_static_8() -> torch.Tensor:
    _, centers = get_turbo_sigmas_and_timesteps(
        num_inference_steps=8,
        use_dynamic_shifting=False,
    )
    return centers.float()


def _voronoi_deltas(centers: torch.Tensor) -> torch.Tensor:
    """Same half-widths as TimestepSampler._sample_turbo_prior."""
    n = centers.numel()
    deltas = torch.empty(n, dtype=centers.dtype, device=centers.device)
    if n == 1:
        deltas[0] = 0.0
        return deltas
    deltas[0] = (centers[0] - centers[1]) * 0.5
    deltas[-1] = (centers[-2] - centers[-1]) * 0.5
    for i in range(1, n - 1):
        d_prev = (centers[i - 1] - centers[i]) * 0.5
        d_next = (centers[i] - centers[i + 1]) * 0.5
        deltas[i] = torch.minimum(d_prev, d_next)
    return deltas


def _train_config(
    *,
    turbo_t_jitter: float = 0.0,
    content_or_style: str = "balanced",
    steps: int = 1000,
    turbo_t_jitter_end=None,
):
    cfg = SimpleNamespace(
        noise_scheduler="flowmatch",
        timestep_type="turbo_prior",
        turbo_prior_steps=8,
        turbo_t_jitter=turbo_t_jitter,
        num_train_timesteps=1000,
        min_denoising_steps=0,
        max_denoising_steps=999,
        content_or_style=content_or_style,
        steps=steps,
    )
    if turbo_t_jitter_end is not None:
        cfg.turbo_t_jitter_end = turbo_t_jitter_end
    return cfg


def _sample(
    jitter: float,
    batch_size: int,
    content_or_style: str = "balanced",
    seed: int = 0,
    *,
    step_num: int = 0,
    steps: int = 1000,
    turbo_t_jitter_end=None,
    turbo_slot_weighting=None,
):
    torch.manual_seed(seed)
    cfg = _train_config(
        turbo_t_jitter=jitter,
        content_or_style=content_or_style,
        steps=steps,
        turbo_t_jitter_end=turbo_t_jitter_end,
    )
    if turbo_slot_weighting is not None:
        cfg.turbo_slot_weighting = turbo_slot_weighting
    sched = SimpleNamespace(timesteps=torch.arange(1000, 0, -1).float())
    sampler = TimestepSampler(cfg, sched)
    latents = torch.zeros(1, 16, 8, 8)
    return sampler.sample(
        batch_size=batch_size,
        latents=latents,
        content_or_style=content_or_style,
        min_noise_steps=0,
        max_noise_steps=999,
        num_train_timesteps=1000,
        device=torch.device("cpu"),
        step_num=step_num,
    )


def _fake_trainer_init(
    self,
    process_id,
    job,
    config,
    *,
    content_or_style: str = "balanced",
    **kwargs,
):
    self.config = config
    self.progress_bar = None
    self.print = lambda *a, **k: None
    mk = dict((config.get("model", {}) or {}).get("model_kwargs", {}) or {})
    self.model_config = SimpleNamespace(model_kwargs=mk)
    self.train_config = SimpleNamespace(
        noise_scheduler="placeholder",
        num_train_timesteps=None,
        loss_type=None,
        timestep_type="turbo_prior",
        content_or_style=content_or_style,
        linear_timesteps=False,
        linear_timesteps2=False,
        snr_gamma=1.0,
        min_snr_gamma=1.0,
        dtype="bf16",
        do_prior_divergence=False,
        timestep_weighting="none",
        train_turbo=False,
        turbo_prior_steps=8,
        turbo_t_jitter=0.5,
    )


def _patch_diffusion_trainer_init(monkeypatch, *, content_or_style: str = "balanced"):
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

    def _init(self, process_id, job, config, **kwargs):
        _fake_trainer_init(
            self, process_id, job, config, content_or_style=content_or_style, **kwargs
        )

    monkeypatch.setattr(DiffusionTrainer, "__init__", _init)


# --- dsigma weights ---


def test_dsigma_weights_last_heaviest_sum_one():
    sigmas, _ = get_turbo_sigmas_and_timesteps(
        num_inference_steps=8,
        use_dynamic_shifting=False,
    )
    next_sigma = torch.cat([sigmas[1:], sigmas.new_zeros(1)])
    expected = (sigmas - next_sigma).abs()
    expected = expected / expected.sum()

    w = turbo_slot_dsigma_weights(8)
    assert w.numel() == 8
    assert torch.allclose(w, expected, atol=1e-6)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)
    assert float(w[-1]) == float(w.max())
    # Last Δσ = σ_last - 0 ≈ 0.30 of total mass on static shift.
    assert float(w[-1]) > 0.25


# --- sampling weights (content / style) ---


def test_sampling_weights_content_is_flipped_dsigma():
    d = turbo_slot_dsigma_weights(8)
    wb = turbo_slot_sampling_weights(8, "balanced")
    ws = turbo_slot_sampling_weights(8, "style")
    wc = turbo_slot_sampling_weights(8, "content")
    assert not torch.allclose(wb, d)
    assert torch.allclose(ws, d)
    assert torch.allclose(wc, d.flip(0))
    assert int(wb.argmax().item()) == 4
    assert float(wb[4]) == float(wb.max())
    assert float(ws[-1]) == float(ws.max()) and float(ws[-1]) > 0.25
    assert float(wc[0]) == float(wc.max()) and float(wc[0]) > 0.25
    assert torch.allclose(wb.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(ws.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(wc.sum(), torch.tensor(1.0), atol=1e-5)
    assert (wb > 0).all()


def test_sampling_weights_balanced_peaks_at_nearest_t750():
    centers8 = _centers_static_8()
    assert abs(float(centers8[4]) - TURBO_BALANCED_TARGET_T) < 1.0
    assert turbo_balanced_target_slot(centers8) == 4
    assert int(turbo_slot_sampling_weights(8, "balanced").argmax().item()) == 4

    _, centers4 = get_turbo_sigmas_and_timesteps(
        num_inference_steps=4,
        use_dynamic_shifting=False,
    )
    target4 = turbo_balanced_target_slot(centers4)
    wb4 = turbo_slot_sampling_weights(4, "balanced")
    assert int(wb4.argmax().item()) == target4
    assert torch.allclose(wb4.sum(), torch.tensor(1.0), atol=1e-5)
    assert (wb4 > 0).all()

    w1 = turbo_slot_sampling_weights(1, "balanced")
    assert w1.numel() == 1
    assert torch.allclose(w1, torch.tensor([1.0]), atol=1e-5)


def test_sampling_weights_jitter0_content_prefers_slot0_balanced_prefers_t750():
    centers = _centers_static_8()
    center_keys = [round(float(x), 4) for x in centers.tolist()]
    want = set(center_keys)
    slot0, slot4, slot7 = center_keys[0], center_keys[4], center_keys[-1]

    def _counts(mode: str):
        result = _sample(
            jitter=0.0,
            batch_size=4096,
            content_or_style=mode,
            seed=0,
        )
        keys = [round(float(x), 4) for x in result.timesteps.tolist()]
        assert set(keys) == want
        return keys.count(slot0), keys.count(slot4), keys.count(slot7)

    c0, c4, c7 = _counts("content")
    assert c0 > c7
    b0, b4, b7 = _counts("balanced")
    assert b4 > b0 and b4 > b7
    s0, s4, s7 = _counts("style")
    assert s7 > s0


# --- turbo_slot_weighting ---


def test_omitted_turbo_slot_weighting_uses_dsigma():
    """No turbo_slot_weighting key → dsigma multinomial (jitter0 hits all centers)."""
    result = _sample(jitter=0.0, batch_size=256, seed=0)
    centers = _centers_static_8()
    got = {round(float(x), 4) for x in result.timesteps.tolist()}
    want = {round(float(x), 4) for x in centers.tolist()}
    assert got == want


def test_explicit_dsigma_turbo_slot_weighting_ok():
    result = _sample(
        jitter=0.0, batch_size=64, seed=0, turbo_slot_weighting="dsigma"
    )
    centers = _centers_static_8()
    got = {round(float(x), 4) for x in result.timesteps.tolist()}
    want = {round(float(x), 4) for x in centers.tolist()}
    assert got.issubset(want)


@pytest.mark.parametrize("bad", ["uniform", "other", "mse"])
def test_non_dsigma_turbo_slot_weighting_raises(bad):
    torch.manual_seed(0)
    cfg = _train_config(turbo_t_jitter=0.0)
    cfg.turbo_slot_weighting = bad
    sched = SimpleNamespace(timesteps=torch.arange(1000, 0, -1).float())
    sampler = TimestepSampler(cfg, sched)
    latents = torch.zeros(1, 16, 8, 8)
    with pytest.raises(ValueError, match="dsigma"):
        sampler.sample(
            batch_size=4,
            latents=latents,
            content_or_style="balanced",
            min_noise_steps=0,
            max_noise_steps=999,
            num_train_timesteps=1000,
            device=torch.device("cpu"),
            step_num=0,
        )


# --- jitter anneal ---


def test_jitter_anneal_default_end_zero_at_last_step():
    """Omitted turbo_t_jitter_end defaults to 0 → last step has zero jitter."""
    steps = 100
    result = _sample(
        jitter=0.5,
        batch_size=256,
        seed=3,
        step_num=steps - 1,
        steps=steps,
        # turbo_t_jitter_end omitted → getattr default 0.0
    )
    centers = _centers_static_8()
    got = {round(float(x), 4) for x in result.timesteps.tolist()}
    want = {round(float(x), 4) for x in centers.tolist()}
    assert got == want


def test_jitter_anneal_step0_uses_start():
    """step_num=0 → effective jitter = start (0.5); samples leave exact centers."""
    steps = 100
    centers = _centers_static_8()
    deltas = _voronoi_deltas(centers)
    result = _sample(
        jitter=0.5,
        batch_size=2048,
        seed=4,
        step_num=0,
        steps=steps,
        turbo_t_jitter_end=0.0,
    )
    ts = result.timesteps.float()
    center_set = {round(float(x), 4) for x in centers.tolist()}
    got = {round(float(x), 4) for x in ts.tolist()}
    assert got != center_set, "step 0 with start=0.5 must apply jitter"

    max_offset = 0.5 * 2.0 * deltas
    dists = (ts.unsqueeze(1) - centers.unsqueeze(0)).abs()
    in_cell = (dists <= max_offset.unsqueeze(0) + 1e-3).any(dim=1)
    assert bool(in_cell.all())


def test_jitter_anneal_end_equals_start_constant():
    """end == start → same non-zero jitter at step 0 and last step."""
    steps = 50
    centers = _centers_static_8()
    center_set = {round(float(x), 4) for x in centers.tolist()}

    r0 = _sample(
        jitter=0.5,
        batch_size=512,
        seed=5,
        step_num=0,
        steps=steps,
        turbo_t_jitter_end=0.5,
    )
    r_last = _sample(
        jitter=0.5,
        batch_size=512,
        seed=5,
        step_num=steps - 1,
        steps=steps,
        turbo_t_jitter_end=0.5,
    )
    assert {round(float(x), 4) for x in r0.timesteps.tolist()} != center_set
    assert {round(float(x), 4) for x in r_last.timesteps.tolist()} != center_set
    # Same seed + same effective jitter → identical samples.
    assert torch.allclose(r0.timesteps, r_last.timesteps)


# --- grid / voronoi (P1 retained) ---


def test_static_8_centers_match_helper_and_jitter0_set():
    centers = _centers_static_8()
    expected = torch.tensor(EXPECTED_CENTERS, dtype=torch.float32)
    assert torch.allclose(centers, expected, atol=1.0)

    result = _sample(jitter=0.0, batch_size=256, seed=0)
    assert result.timestep_indices is None
    got = {round(float(x), 4) for x in result.timesteps.tolist()}
    want = {round(float(x), 4) for x in centers.tolist()}
    assert got == want


def test_jitter_stays_in_voronoi_cells_and_last_slot_not_toward_zero():
    centers = _centers_static_8()
    deltas = _voronoi_deltas(centers)
    # Last-slot floor with j=1: center - 2*delta (must stay well above 0).
    last_floor = float(centers[-1] - 2.0 * deltas[-1])
    assert last_floor > 50.0

    for jitter in (0.5, 1.0):
        # end=start so anneal does not collapse jitter at step 0 with default end=0
        # (default end=0 is fine at step_num=0: progress=0 → start).
        result = _sample(jitter=jitter, batch_size=2048, seed=1)
        ts = result.timesteps.float()
        max_offset = jitter * 2.0 * deltas
        dists = (ts.unsqueeze(1) - centers.unsqueeze(0)).abs()
        in_cell = (dists <= max_offset.unsqueeze(0) + 1e-3).any(dim=1)
        assert bool(in_cell.all()), f"jitter={jitter}: samples escaped Voronoi cells"

        assert float(ts.min()) >= last_floor - 1e-3
        assert float(ts.min()) > 50.0, f"jitter={jitter}: last slot must not extend toward t→0"
        assert float(ts.max()) <= 1000.0 + 1e-3, f"jitter={jitter}: t must not exceed num_train_timesteps"

        mean_t = float(ts.mean())
        assert mean_t > 500.0, (
            f"jitter={jitter}: mean t={mean_t:.1f} should stay on 8-slot prior (~700+), "
            "not gaussian peak ~120"
        )


def test_jitter_clamped_to_num_train_timesteps():
    """Slot-0 Voronoi jitter can exceed 1000 before clamp; after clamp t stays in [0, ntt]."""
    for jitter in (0.5, 1.0):
        result = _sample(
            jitter=jitter,
            batch_size=2048,
            seed=1,
            turbo_t_jitter_end=jitter,
        )
        ts = result.timesteps.float()
        assert float(ts.min()) >= 0.0
        assert float(ts.max()) <= 1000.0 + 1e-3
        assert float(ts.max()) == pytest.approx(1000.0, abs=1e-3), (
            f"jitter={jitter}: clamp must hit t=1000 on slot 0"
        )
        assert bool((ts < 1000.0 - 1e-3).any()), (
            f"jitter={jitter}: some samples must still leave exact centers"
        )


def test_gaussian_content_or_style_does_not_override_turbo_prior():
    result = _sample(
        jitter=0.0,
        batch_size=128,
        content_or_style="gaussian",
        seed=2,
    )
    assert result.timestep_indices is None
    centers = _centers_static_8()
    got = {round(float(x), 4) for x in result.timesteps.tolist()}
    want = {round(float(x), 4) for x in centers.tolist()}
    assert got == want


# --- trainer contracts (P3) ---


def test_turbo_prior_omitted_encoding_defaults_true(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(monkeypatch)
    cfg = {
        "model": {
            "model_kwargs": {
                "use_diffsynth_training_loop": False,
            }
        }
    }
    trainer = ZImageDiffSynthTrainer(0, None, cfg)
    assert trainer.model_config.model_kwargs.get("use_diffsynth_prompt_encoding") is True
    assert cfg["model"]["model_kwargs"]["use_diffsynth_prompt_encoding"] is True


def test_turbo_prior_explicit_encoding_true_ok(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(monkeypatch)
    cfg = {
        "model": {
            "model_kwargs": {
                "use_diffsynth_training_loop": False,
                "use_diffsynth_prompt_encoding": True,
            }
        }
    }
    trainer = ZImageDiffSynthTrainer(0, None, cfg)
    assert trainer.model_config.model_kwargs.get("use_diffsynth_prompt_encoding") is True


def test_turbo_prior_raises_on_encoding_false(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(monkeypatch)
    cfg = {
        "model": {
            "model_kwargs": {
                "use_diffsynth_training_loop": False,
                "use_diffsynth_prompt_encoding": False,
            }
        }
    }
    with pytest.raises(ValueError, match="use_diffsynth_prompt_encoding"):
        ZImageDiffSynthTrainer(0, None, cfg)


def test_turbo_prior_raises_on_diffsynth_training_loop(monkeypatch):
    """timestep_type=turbo_prior + use_diffsynth_training_loop=true → raise."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(monkeypatch)
    cfg = {
        "model": {
            "model_kwargs": {
                "use_diffsynth_training_loop": True,
            }
        }
    }
    with pytest.raises(ValueError, match="use_diffsynth_training_loop"):
        ZImageDiffSynthTrainer(0, None, cfg)


@pytest.mark.parametrize("mode", ["gaussian", "gaussian_bimodal"])
def test_turbo_prior_raises_on_gaussian_content_or_style(monkeypatch, mode):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(monkeypatch, content_or_style=mode)
    cfg = {
        "model": {
            "model_kwargs": {
                "use_diffsynth_training_loop": False,
            }
        }
    }
    with pytest.raises(ValueError, match="gaussian"):
        ZImageDiffSynthTrainer(0, None, cfg)


def test_turbo_prior_raises_on_dynamic_shifting(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(monkeypatch)
    cfg = {
        "model": {
            "model_kwargs": {
                "use_diffsynth_training_loop": False,
                "use_dynamic_shifting": True,
            }
        }
    }
    with pytest.raises(ValueError, match="use_dynamic_shifting"):
        ZImageDiffSynthTrainer(0, None, cfg)
