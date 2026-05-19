"""
Unit tests for DiffSynth-aligned Z-Image helpers (loss scale, no double timestep weight, config parity).

Run from repo root with venv:
  venv\\Scripts\\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_diffsynth_training.py -q
"""

import torch
import torch.nn.functional as F
from types import SimpleNamespace

import pytest

from extensions_built_in.diffusion_models.z_image_diffsynth import diffsynth_training as dst
from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
    _resolve_use_diffsynth_prompt_encoding,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
    ZImageDiffSynthTrainer,
    _read_use_diffsynth_training_loop_from_config,
)


def test_use_diffsynth_prompt_encoding_inherits_training_loop():
    assert _resolve_use_diffsynth_prompt_encoding({}) is True
    assert _resolve_use_diffsynth_prompt_encoding({"use_diffsynth_training_loop": True}) is True
    assert _resolve_use_diffsynth_prompt_encoding({"use_diffsynth_training_loop": False}) is False


def test_use_diffsynth_prompt_encoding_explicit_override():
    assert (
        _resolve_use_diffsynth_prompt_encoding(
            {"use_diffsynth_training_loop": False, "use_diffsynth_prompt_encoding": True}
        )
        is True
    )
    assert (
        _resolve_use_diffsynth_prompt_encoding(
            {"use_diffsynth_training_loop": True, "use_diffsynth_prompt_encoding": False}
        )
        is False
    )


def test_read_use_diffsynth_training_loop_matches_model_kwargs():
    assert _read_use_diffsynth_training_loop_from_config({}) is True
    assert (
        _read_use_diffsynth_training_loop_from_config(
            {"model": {"model_kwargs": {"use_diffsynth_training_loop": False}}}
        )
        is False
    )


def test_aggregate_matches_flow_match_sft_scale_b1():
    """FlowMatchSFTLoss: mse_loss(..., reduction='mean') * training_weight (scalar)."""
    torch.manual_seed(0)
    pred = torch.randn(1, 4, 8, 8)
    target = torch.randn(1, 4, 8, 8)
    timesteps = torch.tensor([100.0])
    w = torch.tensor([1.75])
    mask = torch.ones(1, 1, 8, 8)
    out = dst.aggregate_flow_matching_mse_diffsynth(
        pred,
        target,
        timesteps,
        w,
        mask,
        pred,
        train_turbo=False,
        log_writer=None,
        step_num=0,
        is_main_process=False,
        log_every=None,
    )
    ref = F.mse_loss(pred.float(), target.float()) * w.to(pred.device)
    assert out.shape == (1,)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_aggregate_masked_spatial_mean_then_timestep_weight():
    """Explicit formula: w_b * mean_{C,H,W}((pred-target)² * mask)."""
    pred = torch.zeros(1, 1, 2, 2)
    target = torch.ones(1, 1, 2, 2)
    m = torch.zeros(1, 1, 2, 2)
    m[..., 0, 0] = 1.0
    w = torch.tensor([3.0])
    timesteps = torch.tensor([0.0])
    out = dst.aggregate_flow_matching_mse_diffsynth(
        pred,
        target,
        timesteps,
        w,
        m,
        pred,
        train_turbo=False,
        log_writer=None,
        step_num=0,
        is_main_process=False,
        log_every=None,
    )
    # mean(sq * m) = 1/4; times w = 0.75
    assert torch.allclose(out, torch.tensor([0.75]))
    ref = w * F.mse_loss(pred, target, reduction="none").mul(m).mean(dim=(1, 2, 3))
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-6)


def test_zimage_trainer_aggregate_applies_timestep_weight_once():
    class TSched:
        def get_weights_for_timesteps(self, timesteps, v2=False, timestep_type="linear"):
            return torch.full(
                (timesteps.shape[0],),
                2.0,
                device=timesteps.device,
                dtype=torch.float32,
            )

    z = object.__new__(ZImageDiffSynthTrainer)
    z.use_diffsynth_training_loop = True
    z.train_config = SimpleNamespace(
        do_prior_divergence=False,
        train_turbo=False,
        linear_timesteps2=False,
        timestep_type="linear",
    )
    z.sd = SimpleNamespace(noise_scheduler=TSched())
    z.writer = None
    z.step_num = 0
    z.accelerator = SimpleNamespace(is_main_process=False)
    z.logging_config = SimpleNamespace(log_every=None)
    pred = torch.randn(1, 2, 4, 4)
    target = torch.randn(1, 2, 4, 4)
    ts = torch.tensor([100.0])
    mm = torch.ones(1, 1, 4, 4)
    out = ZImageDiffSynthTrainer._aggregate_flow_matching_mse_loss(
        z, pred, target, ts, mm, pred, None, None
    )
    ref = 2.0 * F.mse_loss(pred.float(), target.float())
    assert out.shape == (1,)
    assert torch.allclose(out, ref.view(1), rtol=1e-5, atol=1e-6)


def test_default_aggregate_no_double_timestep_weight():
    """
    Regression: MSE path must apply linear_timesteps_weights only once (via hook), not hook + _apply.
    """
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from types import SimpleNamespace

    calls = {"apply": 0}

    class TSched:
        timesteps = torch.linspace(1000, 0, 1000)

        def get_weights_for_timesteps(self, timesteps, v2=False, timestep_type="linear"):
            return torch.ones(timesteps.shape[0], device=timesteps.device, dtype=torch.float32) * 2.0

    class TAcc:
        is_main_process = False

    class Fake(SDTrainer):
        def __init__(self):
            self.train_config = SimpleNamespace(
                linear_timesteps=True,
                linear_timesteps2=False,
                timestep_type="linear",
                do_prior_divergence=False,
                train_turbo=False,
                loss_type="mse",
            )
            self.sd = SimpleNamespace(is_flow_matching=True, noise_scheduler=TSched())
            self.writer = None
            self.step_num = 0
            self.logging_config = SimpleNamespace(log_every=None)
            self.accelerator = TAcc()

        def _apply_flow_timestep_element_weights(self, loss, timesteps):
            calls["apply"] += 1
            return super()._apply_flow_timestep_element_weights(loss, timesteps)

    f = Fake()
    pred = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    ts = torch.tensor([500.0])
    mm = torch.ones(1, 1, 4, 4)
    out = f._aggregate_flow_matching_mse_loss(pred, target, ts, mm, pred, None, None)
    assert out.shape == (1,)
    assert calls["apply"] == 1
    # mask all ones, mean sq = 0, loss = 0
    assert float(out.item()) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
