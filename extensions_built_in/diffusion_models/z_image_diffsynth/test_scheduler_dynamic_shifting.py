"""
Tests for model.model_kwargs.use_dynamic_shifting (Flux-style dynamic time shifting).

Run from repo root:
  venv\\Scripts\\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_scheduler_dynamic_shifting.py -q
"""

import warnings

import torch
import pytest

from extensions_built_in.diffusion_models.z_image_diffsynth.model import ZImageDiffSynthModel
from extensions_built_in.diffusion_models.z_image_diffsynth.scheduler_config import (
    build_scheduler_config,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.scheduler_adapter import (
    DiffSynthZImageSchedulerAdapter,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.sampling import (
    _get_diffsynth_scheduler,
)
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


def test_build_scheduler_config_static():
    cfg = build_scheduler_config(False)
    assert cfg["use_dynamic_shifting"] is False
    assert "base_image_seq_len" not in cfg
    assert cfg["shift"] == 3.0


def test_build_scheduler_config_dynamic():
    cfg = build_scheduler_config(True)
    assert cfg["use_dynamic_shifting"] is True
    assert cfg["base_image_seq_len"] == 256
    assert cfg["max_shift"] == 1.15


def test_get_train_scheduler_dynamic_is_custom_flowmatch():
    sched = ZImageDiffSynthModel.get_train_scheduler(
        use_diffsynth_loop=False,
        use_dynamic_shifting=True,
    )
    assert isinstance(sched, CustomFlowMatchEulerDiscreteScheduler)
    assert not isinstance(sched, DiffSynthZImageSchedulerAdapter)
    assert sched.config.use_dynamic_shifting is True


def test_get_train_scheduler_diffsynth_loop_unchanged():
    sched = ZImageDiffSynthModel.get_train_scheduler(
        use_diffsynth_loop=True,
        use_dynamic_shifting=False,
    )
    assert isinstance(sched, DiffSynthZImageSchedulerAdapter)


def _shift_timesteps_for_latents(sched, h, w):
    latents = torch.zeros(1, 16, h, w)
    sched.set_train_timesteps(
        1000,
        device="cpu",
        timestep_type="shift",
        latents=latents,
        patch_size=2,
    )
    return sched.timesteps.clone()


def test_dynamic_shift_timesteps_vary_by_resolution():
    sched = ZImageDiffSynthModel.get_train_scheduler(
        use_diffsynth_loop=False,
        use_dynamic_shifting=True,
    )
    t_small = _shift_timesteps_for_latents(sched, 64, 64)
    t_large = _shift_timesteps_for_latents(sched, 128, 128)
    assert not torch.allclose(t_small, t_large)


def test_static_shift_timesteps_same_across_resolutions():
    sched = ZImageDiffSynthModel.get_train_scheduler(
        use_diffsynth_loop=False,
        use_dynamic_shifting=False,
    )
    t_small = _shift_timesteps_for_latents(sched, 64, 64)
    t_large = _shift_timesteps_for_latents(sched, 128, 128)
    assert torch.allclose(t_small, t_large)


def test_dynamic_shift_timesteps_no_divide_by_zero_warning():
    sched = ZImageDiffSynthModel.get_train_scheduler(
        use_diffsynth_loop=False,
        use_dynamic_shifting=True,
    )
    latents = torch.zeros(1, 16, 64, 64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        sched.set_train_timesteps(
            1000,
            device="cpu",
            timestep_type="shift",
            latents=latents,
            patch_size=2,
        )
    divide_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning) and "divide by zero" in str(w.message)
    ]
    assert divide_warnings == []


def test_sampling_dynamic_vs_static_scheduler():
    sig_dyn, _ = _get_diffsynth_scheduler(
        32,
        use_dynamic_shifting=True,
        latent_h=64,
        latent_w=64,
    )
    sig_dyn_large, _ = _get_diffsynth_scheduler(
        32,
        use_dynamic_shifting=True,
        latent_h=128,
        latent_w=128,
    )
    sig_static, _ = _get_diffsynth_scheduler(
        32,
        use_dynamic_shifting=False,
        latent_h=64,
        latent_w=64,
    )
    sig_static_large, _ = _get_diffsynth_scheduler(
        32,
        use_dynamic_shifting=False,
        latent_h=128,
        latent_w=128,
    )
    assert not torch.allclose(sig_dyn, sig_dyn_large)
    assert torch.allclose(sig_static, sig_static_large)
    assert not torch.allclose(sig_dyn, sig_static)
