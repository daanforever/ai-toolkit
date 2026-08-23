"""
CPU unit tests for timestep_type=turbo_prior (Turbo 8-NFE grid + Voronoi jitter).

No model weights. Style mirrors test_index_timestep_contract.py.

Run from repo root:
  venv\\Scripts\\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_turbo_prior.py -q
"""

from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models.z_image_diffsynth.turbo_schedule import (
    get_turbo_sigmas_and_timesteps,
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


def _train_config(*, turbo_t_jitter: float = 0.0, content_or_style: str = "balanced"):
    return SimpleNamespace(
        noise_scheduler="flowmatch",
        timestep_type="turbo_prior",
        turbo_prior_steps=8,
        turbo_t_jitter=turbo_t_jitter,
        num_train_timesteps=1000,
        min_denoising_steps=0,
        max_denoising_steps=999,
        content_or_style=content_or_style,
    )


def _sample(jitter: float, batch_size: int, content_or_style: str = "balanced", seed: int = 0):
    torch.manual_seed(seed)
    sched = SimpleNamespace(timesteps=torch.arange(1000, 0, -1).float())
    sampler = TimestepSampler(_train_config(turbo_t_jitter=jitter, content_or_style=content_or_style), sched)
    latents = torch.zeros(1, 16, 8, 8)
    return sampler.sample(
        batch_size=batch_size,
        latents=latents,
        content_or_style=content_or_style,
        min_noise_steps=0,
        max_noise_steps=999,
        num_train_timesteps=1000,
        device=torch.device("cpu"),
        step_num=0,
    )


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
        result = _sample(jitter=jitter, batch_size=2048, seed=1)
        ts = result.timesteps.float()
        # Every sample must land in some closed sampling cell.
        # Cell i: |t - c_i| <= jitter * 2 * delta_i (+ tiny float slack).
        max_offset = jitter * 2.0 * deltas
        dists = (ts.unsqueeze(1) - centers.unsqueeze(0)).abs()
        in_cell = (dists <= max_offset.unsqueeze(0) + 1e-3).any(dim=1)
        assert bool(in_cell.all()), f"jitter={jitter}: samples escaped Voronoi cells"

        assert float(ts.min()) >= last_floor - 1e-3
        assert float(ts.min()) > 50.0, f"jitter={jitter}: last slot must not extend toward t→0"

        mean_t = float(ts.mean())
        assert mean_t > 500.0, (
            f"jitter={jitter}: mean t={mean_t:.1f} should stay on 8-slot prior (~700+), "
            "not gaussian peak ~120"
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


def test_turbo_prior_ignores_diffsynth_training_loop(monkeypatch):
    """timestep_type=turbo_prior + use_diffsynth_training_loop=true → loop off, type kept."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

    def _fake_init(self, process_id, job, config, **kwargs):
        self.config = config
        self.progress_bar = None
        mk = dict((config.get("model", {}) or {}).get("model_kwargs", {}) or {})
        self.model_config = SimpleNamespace(model_kwargs=mk)
        self.train_config = SimpleNamespace(
            noise_scheduler="placeholder",
            num_train_timesteps=None,
            loss_type=None,
            timestep_type="turbo_prior",
            content_or_style="gaussian",
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

    monkeypatch.setattr(DiffusionTrainer, "__init__", _fake_init)
    cfg = {
        "model": {
            "model_kwargs": {
                "use_diffsynth_training_loop": True,
            }
        }
    }
    trainer = ZImageDiffSynthTrainer(0, None, cfg)

    assert trainer.use_diffsynth_training_loop is False
    assert trainer.train_config.timestep_type == "turbo_prior"
