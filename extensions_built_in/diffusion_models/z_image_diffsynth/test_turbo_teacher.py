"""
CPU unit tests for train.turbo_teacher_weight (Turbo-teacher consistency).

No real model weights. Run from repo root:
  venv\\Scripts\\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_turbo_teacher.py -q
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from toolkit.config_modules import TrainConfig


def _fake_trainer_init(
    self,
    process_id,
    job,
    config,
    *,
    content_or_style: str = "balanced",
    timestep_type: str = "turbo_prior",
    turbo_teacher_weight: float = 0.0,
    sampling_name_or_path=None,
    **kwargs,
):
    self.config = config
    self.progress_bar = None
    self.print = lambda *a, **k: None
    mk = dict((config.get("model", {}) or {}).get("model_kwargs", {}) or {})
    sampling = sampling_name_or_path
    if sampling is None:
        sampling = (config.get("model", {}) or {}).get("sampling_name_or_path")
    self.model_config = SimpleNamespace(
        model_kwargs=mk,
        sampling_name_or_path=sampling,
        low_vram=False,
    )
    self.train_config = SimpleNamespace(
        noise_scheduler="placeholder",
        num_train_timesteps=None,
        loss_type=None,
        timestep_type=timestep_type,
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
        turbo_teacher_weight=turbo_teacher_weight,
        pred_scaler=1.0,
        match_noise_norm=False,
    )
    self.writer = None
    self.logger = MagicMock()
    self.accelerator = SimpleNamespace(is_main_process=True)
    self.step_num = 0
    self._turbo_teacher_embeds = None
    self._turbo_teacher_pred = None
    self.sd = SimpleNamespace(_sampling_transformer=None)


def _patch_diffusion_trainer_init(
    monkeypatch,
    *,
    content_or_style: str = "balanced",
    timestep_type: str = "turbo_prior",
    turbo_teacher_weight: float = 0.0,
    sampling_name_or_path=None,
):
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

    def _init(self, process_id, job, config, **kwargs):
        _fake_trainer_init(
            self,
            process_id,
            job,
            config,
            content_or_style=content_or_style,
            timestep_type=timestep_type,
            turbo_teacher_weight=turbo_teacher_weight,
            sampling_name_or_path=sampling_name_or_path,
            **kwargs,
        )

    monkeypatch.setattr(DiffusionTrainer, "__init__", _init)


def _base_cfg(*, use_diffsynth_training_loop: bool = False, sampling_path="/tmp/turbo"):
    return {
        "model": {
            "sampling_name_or_path": sampling_path,
            "model_kwargs": {
                "use_diffsynth_training_loop": use_diffsynth_training_loop,
            },
        }
    }


# --- TrainConfig default ---


def test_train_config_turbo_teacher_weight_default_zero():
    tc = TrainConfig(**{})
    assert float(tc.turbo_teacher_weight) == 0.0


def test_train_config_turbo_teacher_weight_set():
    tc = TrainConfig(**{"turbo_teacher_weight": 0.25})
    assert float(tc.turbo_teacher_weight) == 0.25


# --- Init raise contracts ---


def test_w_gt0_raises_without_turbo_prior(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(
        monkeypatch,
        timestep_type="linear",
        turbo_teacher_weight=0.25,
        sampling_name_or_path="/tmp/turbo",
    )
    with pytest.raises(ValueError, match="turbo_prior"):
        ZImageDiffSynthTrainer(0, None, _base_cfg())


def test_w_gt0_raises_on_diffsynth_training_loop(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=0.25,
        sampling_name_or_path="/tmp/turbo",
    )
    with pytest.raises(ValueError, match="use_diffsynth_training_loop"):
        ZImageDiffSynthTrainer(
            0, None, _base_cfg(use_diffsynth_training_loop=True)
        )


def test_w_gt0_raises_without_sampling_path(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=0.25,
        sampling_name_or_path=None,
    )
    cfg = {
        "model": {
            "model_kwargs": {"use_diffsynth_training_loop": False},
        }
    }
    with pytest.raises(ValueError, match="sampling_name_or_path"):
        ZImageDiffSynthTrainer(0, None, cfg)


def test_w_gt0_ok_with_contracts(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=0.25,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())
    assert float(trainer.train_config.turbo_teacher_weight) == 0.25


# --- w=0 no-op / loss path ---


def test_w_zero_skips_teacher_forward(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=0.0,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())

    called = {"n": 0}

    def _fake_super_calc(self, *args, **kwargs):
        return torch.tensor(1.5, requires_grad=True)

    def _fake_teacher(*args, **kwargs):
        called["n"] += 1
        return torch.zeros(1, 4, 2, 2)

    def _fake_super_pred(self, *args, **kwargs):
        return torch.zeros(1, 4, 2, 2, requires_grad=True)

    monkeypatch.setattr(SDTrainer, "calculate_loss", _fake_super_calc)
    monkeypatch.setattr(SDTrainer, "predict_noise", _fake_super_pred)
    trainer.sd.get_turbo_teacher_prediction = _fake_teacher
    trainer.sd._sampling_transformer = nn.Linear(1, 1)

    from toolkit.prompt_utils import PromptEmbeds

    embeds = PromptEmbeds(torch.randn(1, 4, 8))
    trainer.predict_noise(
        noisy_latents=torch.randn(1, 4, 2, 2),
        timesteps=torch.tensor([500.0]),
        conditional_embeds=embeds,
        batch=SimpleNamespace(prompt_embeds=embeds),
        is_primary_pred=True,
    )
    assert called["n"] == 0

    noise_pred = torch.randn(1, 4, 2, 2, requires_grad=True)
    out = trainer.calculate_loss(
        noise_pred=noise_pred,
        noise=torch.randn(1, 4, 2, 2),
        noisy_latents=torch.randn(1, 4, 2, 2),
        timesteps=torch.tensor([500.0]),
        batch=SimpleNamespace(prompt_embeds=None),
        mask_multiplier=1.0,
        prior_pred=None,
    )
    assert called["n"] == 0
    assert float(out.detach()) == pytest.approx(1.5)


def test_w_gt0_adds_turbo_mse_and_logs(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from toolkit.prompt_utils import PromptEmbeds

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=0.5,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())

    def _fake_super_calc(self, *args, **kwargs):
        return torch.tensor(2.0, requires_grad=True)

    def _fake_super_pred(self, *args, **kwargs):
        return torch.zeros(1, 4, 2, 2, requires_grad=True)

    turbo_param = nn.Parameter(torch.randn(2, 2), requires_grad=False)
    fake_mod = nn.Linear(1, 1, bias=False)
    for p in fake_mod.parameters():
        p.requires_grad_(False)

    def _fake_teacher(latent, timestep, embeds, batch=None, **kwargs):
        _ = turbo_param * 0
        return torch.ones(1, 4, 2, 2)

    monkeypatch.setattr(SDTrainer, "calculate_loss", _fake_super_calc)
    monkeypatch.setattr(SDTrainer, "predict_noise", _fake_super_pred)
    trainer.sd.get_turbo_teacher_prediction = _fake_teacher
    trainer.sd._sampling_transformer = fake_mod

    embeds = PromptEmbeds(torch.randn(1, 4, 8))
    trainer.predict_noise(
        noisy_latents=torch.randn(1, 4, 2, 2),
        timesteps=torch.tensor([500.0]),
        conditional_embeds=embeds,
        batch=SimpleNamespace(prompt_embeds=embeds),
        is_primary_pred=True,
    )
    assert trainer._turbo_teacher_pred is not None

    student = torch.zeros(1, 4, 2, 2, requires_grad=True)
    out = trainer.calculate_loss(
        noise_pred=student,
        noise=torch.randn(1, 4, 2, 2),
        noisy_latents=torch.randn(1, 4, 2, 2),
        timesteps=torch.tensor([500.0]),
        batch=SimpleNamespace(prompt_embeds=embeds),
        mask_multiplier=1.0,
        prior_pred=None,
    )
    # L_fm=2, L_turbo=MSE(0,1)=1, w=0.5 → 2 + 0.5*1 = 2.5
    assert torch.isfinite(out)
    assert float(out.detach()) == pytest.approx(2.5)
    assert trainer._turbo_teacher_pred is None  # cleared after use
    log_calls = [c.args[0] for c in trainer.logger.log.call_args_list if c.args]
    assert any(
        isinstance(d, dict) and "loss/turbo_teacher" in d and abs(d["loss/turbo_teacher"] - 1.0) < 1e-5
        for d in log_calls
    )

    out.backward()
    assert student.grad is not None
    assert turbo_param.grad is None
    for p in fake_mod.parameters():
        assert p.grad is None
        assert p.requires_grad is False


def test_w_gt0_missing_sampling_transformer_raises(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from toolkit.prompt_utils import PromptEmbeds

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=0.25,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())
    monkeypatch.setattr(
        SDTrainer, "predict_noise", lambda *a, **k: torch.zeros(1, 4, 2, 2)
    )
    trainer.sd._sampling_transformer = None
    embeds = PromptEmbeds(torch.randn(1, 4, 8))
    with pytest.raises(ValueError, match="sampling transformer"):
        trainer.predict_noise(
            noisy_latents=torch.zeros(1, 4, 2, 2),
            timesteps=torch.tensor([500.0]),
            conditional_embeds=embeds,
            batch=SimpleNamespace(prompt_embeds=embeds),
            is_primary_pred=True,
        )


def test_w_gt0_calculate_loss_requires_precomputed_pred(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=0.25,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())
    monkeypatch.setattr(
        SDTrainer, "calculate_loss", lambda *a, **k: torch.tensor(1.0)
    )
    trainer._turbo_teacher_pred = None
    with pytest.raises(ValueError, match="precomputed Turbo teacher"):
        trainer.calculate_loss(
            noise_pred=torch.zeros(1, 4, 2, 2, requires_grad=True),
            noise=torch.zeros(1, 4, 2, 2),
            noisy_latents=torch.zeros(1, 4, 2, 2),
            timesteps=torch.tensor([500.0]),
            batch=SimpleNamespace(prompt_embeds=None),
            mask_multiplier=1.0,
            prior_pred=None,
        )


def test_get_turbo_teacher_prediction_exclusive_offload_order(monkeypatch):
    """Exclusive residency: place-main-CPU → sampling-CUDA → restore before return."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )
    from toolkit.prompt_utils import PromptEmbeds

    class _FakeDit(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1), requires_grad=False)

        def forward(self, latent_list, timestep, text_embeds, return_dict=False):
            b = len(latent_list)
            h, w = latent_list[0].shape[-2], latent_list[0].shape[-1]
            c = latent_list[0].shape[0]
            outs = [torch.ones(c, 1, h, w) * self.w for _ in range(b)]
            return (outs,)

    order = []

    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.train_torch_dtype = torch.float32
    model.model_config = SimpleNamespace(low_vram=False)
    model._sampling_is_diffusers = True
    model.network = None
    dit = _FakeDit()
    wrapper = SimpleNamespace(_inner_dit=dit)
    model._sampling_transformer = wrapper

    def _place(device):
        order.append(("place_main", str(device)))

    def _move_sampling(device):
        order.append(("move_sampling", str(device)))

    def _move_main(device):
        order.append(("restore_main", str(device)))

    def _flush():
        order.append(("flush",))

    model._place_training_dit = _place
    model._move_sampling_transformer = _move_sampling
    model._move_main_network = _move_main
    model._flush_cuda = _flush

    latents = torch.randn(1, 16, 4, 4)
    embeds = PromptEmbeds(torch.randn(1, 8, 16))
    pred = ZImageDiffSynthModel.get_turbo_teacher_prediction(
        model,
        latents,
        torch.tensor([500.0]),
        embeds,
    )
    assert torch.isfinite(pred).all()
    # offload main → flush → sampling on → (forward) → sampling cpu via restore → restore main → flush
    assert order[0] == ("place_main", "cpu")
    assert ("flush",) in order
    assert ("move_sampling", "cpu") in order or any(
        c[0] == "move_sampling" and "cpu" in c[1] for c in order
    )
    # First sampling move after place_main should be onto train device (cpu here)
    sampling_moves = [c for c in order if c[0] == "move_sampling"]
    assert sampling_moves[0][0] == "move_sampling"
    assert order[-2][0] == "restore_main" or order[-1][0] == "flush"
    assert any(c[0] == "restore_main" for c in order)
    # place_main before first sampling move
    place_idx = next(i for i, c in enumerate(order) if c[0] == "place_main")
    samp_idx = next(i for i, c in enumerate(order) if c[0] == "move_sampling")
    assert place_idx < samp_idx


def test_get_turbo_teacher_prediction_no_grad_into_turbo(monkeypatch):
    """Helper forward must not populate .grad on sampling DiT params."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )
    from toolkit.prompt_utils import PromptEmbeds

    class _FakeDit(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1), requires_grad=False)

        def forward(self, latent_list, timestep, text_embeds, return_dict=False):
            # Produce CFHW list matching Diffusers Z-Image out shape
            b = len(latent_list)
            h, w = latent_list[0].shape[-2], latent_list[0].shape[-1]
            c = latent_list[0].shape[0]
            outs = [torch.ones(c, 1, h, w) * self.w for _ in range(b)]
            return (outs,)

    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.train_torch_dtype = torch.float32
    model.model_config = SimpleNamespace(low_vram=False)
    model._sampling_is_diffusers = True
    model.network = None
    dit = _FakeDit()
    wrapper = SimpleNamespace(_inner_dit=dit)
    model._sampling_transformer = wrapper
    model._place_training_dit = lambda device: None
    model._move_sampling_transformer = lambda device: None
    model._move_main_network = lambda device: None
    model._flush_cuda = lambda: None

    latents = torch.randn(1, 16, 4, 4, requires_grad=True)
    embeds = PromptEmbeds(torch.randn(1, 8, 16))
    pred = ZImageDiffSynthModel.get_turbo_teacher_prediction(
        model,
        latents,
        torch.tensor([500.0]),
        embeds,
    )
    assert torch.isfinite(pred).all()
    assert pred.shape == latents.shape
    # Negate convention: ones → -ones after Diffusers path
    assert torch.allclose(pred, -torch.ones_like(pred))
    loss = (latents - pred.detach()).pow(2).mean()
    loss.backward()
    assert dit.w.grad is None
    assert dit.w.requires_grad is False
