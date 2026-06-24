"""
Unit tests for DiffSynth-aligned Z-Image helpers (loss scale, no double timestep weight, config parity).

Run from repo root with venv:
  venv\\Scripts\\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_diffsynth_training.py -q
"""

import torch
import torch.nn.functional as F
from types import SimpleNamespace
import importlib

import pytest

from extensions_built_in.diffusion_models.z_image_diffsynth import diffsynth_training as dst
from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
    _resolve_use_diffsynth_prompt_encoding,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
    ZImageDiffSynthTrainer,
    _read_use_diffsynth_training_loop_from_config,
)
from toolkit.config_modules import TrainConfig


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


def test_aggregate_train_turbo_slices_mask_channels_and_interpolates():
    """train_turbo: mask[:, 3:] then nearest resize to pred spatial size (SDTrainer parity)."""
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    m = torch.zeros(1, 6, 8, 8)
    m[:, 0, 0, 0] = 999.0  # ignored: only channels from index 3 are used
    m[:, 3, 4, 4] = 1.0
    w = torch.tensor([2.0])
    timesteps = torch.tensor([0.0])
    out = dst.aggregate_flow_matching_mse_diffsynth(
        pred,
        target,
        timesteps,
        w,
        m,
        pred,
        train_turbo=True,
        log_writer=None,
        step_num=0,
        is_main_process=False,
        log_every=None,
    )
    mm = m[:, 3:, :, :]
    mm = F.interpolate(mm, size=(pred.shape[2], pred.shape[3]), mode="nearest")
    ref = w * F.mse_loss(pred, target, reduction="none").mul(mm).mean(dim=(1, 2, 3))
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-6)
    assert float(out.item()) < 1.0


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
        timestep_weighting="none",
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


def test_default_aggregate_does_not_apply_timestep_weight():
    """
    Regression: SDTrainer aggregate path should not apply timestep weighting directly.
    Weighting is applied later in calculate_loss after SNR.
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
    assert calls["apply"] == 0
    # mask all ones, mean sq = 0, loss = 0
    assert float(out.item()) == 0.0


def test_calculate_loss_applies_timestep_weight_after_snr(monkeypatch):
    sdtrainer_module = importlib.import_module("extensions_built_in.sd_trainer.SDTrainer")
    SDTrainer = sdtrainer_module.SDTrainer

    class _NoiseScheduler:
        def __init__(self):
            self.timesteps = torch.arange(1000, dtype=torch.float32)
            self.config = SimpleNamespace(
                num_train_timesteps=1000,
                prediction_type="epsilon",
            )

    class _Batch:
        def __init__(self, bsz: int):
            self.mask_tensor = None
            self.loss_multiplier_list = [1.0] * bsz
            self.latents = torch.zeros(bsz, 1, 2, 2)
            self.tensor = torch.zeros_like(self.latents)
            self.file_items = None
            self.audio_pred = None
            self.audio_target = None
            self.sigmas = None

        def get_is_reg_list(self):
            return [False] * self.latents.shape[0]

    trainer = object.__new__(SDTrainer)
    cfg = TrainConfig()
    cfg.dtype = "fp32"
    cfg.loss_target = "noise"
    cfg.loss_type = "mse"
    cfg.train_turbo = False
    cfg.do_guidance_loss = False
    cfg.do_differential_guidance = False
    cfg.do_prior_divergence = False
    cfg.inverted_mask_prior = False
    cfg.correct_pred_norm = False
    cfg.match_noise_norm = False
    cfg.pred_scaler = 1.0
    cfg.target_noise_multiplier = 1.0
    cfg.learnable_snr_gos = False
    cfg.snr_gamma = None
    cfg.min_snr_gamma = 1.0
    cfg.prediction_type = "epsilon"
    cfg.linear_timesteps = False
    cfg.linear_timesteps2 = False
    cfg.timestep_weighting = "gaussian"
    cfg.content_or_style = "balanced"
    cfg.target_norm_std = False
    trainer.train_config = cfg
    trainer.sd = SimpleNamespace(
        is_flow_matching=True,
        prediction_type="epsilon",
        noise_scheduler=_NoiseScheduler(),
    )
    trainer.device_torch = torch.device("cpu")
    trainer.dfe = None
    trainer.adapter = None
    trainer.writer = None
    trainer.logger = None
    trainer.step_num = 0
    trainer.logging_config = SimpleNamespace(log_every=None)
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    trainer.snr_gos = None

    state = {"snr_called": False, "timestep_called": 0}

    def _wrapped_apply_snr_weight(
        loss,
        timesteps,
        noise_scheduler,
        gamma,
        fixed=False,
        prediction_type="epsilon",
    ):
        state["snr_called"] = True
        return loss * 3.0

    def _wrapped_timestep_weight(self, loss, timesteps):
        state["timestep_called"] += 1
        assert state["snr_called"], "timestep weighting must run after SNR weighting"
        assert loss.dim() == 1
        return loss * 5.0

    monkeypatch.setattr(sdtrainer_module, "apply_snr_weight", _wrapped_apply_snr_weight)
    monkeypatch.setattr(SDTrainer, "_apply_flow_timestep_element_weights", _wrapped_timestep_weight)

    bsz = 2
    batch = _Batch(bsz)
    noise_pred = torch.zeros(bsz, 1, 2, 2)
    noise = torch.ones_like(noise_pred)
    noisy_latents = torch.zeros_like(noise_pred)
    timesteps = torch.tensor([100.0, 900.0])

    loss = SDTrainer.calculate_loss(
        trainer,
        noise_pred=noise_pred,
        noise=noise,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        batch=batch,
    )
    assert state["timestep_called"] == 1
    assert torch.allclose(loss, torch.tensor(15.0), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
