"""
CPU unit tests for train.turbo_teacher_weight (bool Turbo train routing).

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
    turbo_teacher_weight: bool = False,
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
    self.sd = SimpleNamespace(_sampling_transformer=None)


def _patch_diffusion_trainer_init(
    monkeypatch,
    *,
    content_or_style: str = "balanced",
    timestep_type: str = "turbo_prior",
    turbo_teacher_weight: bool = False,
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


# --- TrainConfig parse ---


def test_train_config_turbo_teacher_weight_default_false():
    tc = TrainConfig(**{})
    assert tc.turbo_teacher_weight is False


def test_train_config_turbo_teacher_weight_true_false():
    assert TrainConfig(**{"turbo_teacher_weight": True}).turbo_teacher_weight is True
    assert TrainConfig(**{"turbo_teacher_weight": False}).turbo_teacher_weight is False


def test_train_config_turbo_teacher_weight_rejects_float():
    with pytest.raises(ValueError, match="boolean"):
        TrainConfig(**{"turbo_teacher_weight": 0.25})


# --- Init raise contracts ---


def test_true_raises_without_turbo_prior(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(
        monkeypatch,
        timestep_type="linear",
        turbo_teacher_weight=True,
        sampling_name_or_path="/tmp/turbo",
    )
    with pytest.raises(ValueError, match="turbo_prior"):
        ZImageDiffSynthTrainer(0, None, _base_cfg())


def test_true_raises_on_diffsynth_training_loop(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=True,
        sampling_name_or_path="/tmp/turbo",
    )
    with pytest.raises(ValueError, match="use_diffsynth_training_loop"):
        ZImageDiffSynthTrainer(
            0, None, _base_cfg(use_diffsynth_training_loop=True)
        )


def test_true_raises_without_sampling_path(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=True,
        sampling_name_or_path=None,
    )
    cfg = {
        "model": {
            "model_kwargs": {"use_diffsynth_training_loop": False},
        }
    }
    with pytest.raises(ValueError, match="sampling_name_or_path"):
        ZImageDiffSynthTrainer(0, None, cfg)


def test_true_ok_with_contracts(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=True,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())
    assert trainer.train_config.turbo_teacher_weight is True


def test_apply_runtime_turbo_teacher_mode_calls_sd(monkeypatch):
    """Trainer residency flip must invoke sd.apply_turbo_teacher_mode (not hasattr-skip)."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=False,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())
    trainer.use_diffsynth_training_loop = False
    trainer.sd = SimpleNamespace(
        _sampling_transformer=object(),
        apply_turbo_teacher_mode=MagicMock(),
    )

    ZImageDiffSynthTrainer.apply_runtime_turbo_teacher_mode(trainer, True)
    trainer.sd.apply_turbo_teacher_mode.assert_called_with(True)
    assert trainer.train_config.turbo_teacher_weight is True

    ZImageDiffSynthTrainer.apply_runtime_turbo_teacher_mode(trainer, False)
    trainer.sd.apply_turbo_teacher_mode.assert_called_with(False)
    assert trainer.train_config.turbo_teacher_weight is False


def test_apply_runtime_turbo_teacher_weight_calls_mode(monkeypatch):
    """DB bool True via DiffusionTrainer.apply_runtime_turbo_teacher_weight live-swaps."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=False,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())
    trainer.use_diffsynth_training_loop = False
    trainer.is_ui_trainer = True
    trainer._last_applied_runtime_turbo_teacher_weight = None
    trainer.sd = SimpleNamespace(
        _sampling_transformer=object(),
        apply_turbo_teacher_mode=MagicMock(),
    )
    monkeypatch.setattr(
        trainer, "get_runtime_turbo_teacher_weight", lambda: True
    )

    DiffusionTrainer.apply_runtime_turbo_teacher_weight(trainer)
    trainer.sd.apply_turbo_teacher_mode.assert_called_with(True)
    assert trainer.train_config.turbo_teacher_weight is True
    assert trainer._last_applied_runtime_turbo_teacher_weight is True


# --- Mode false: FM-only, no teacher MSE path ---


def test_mode_false_calculate_loss_is_fm_only(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=False,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())

    called = {"teacher": 0}

    def _fake_super_calc(self, *args, **kwargs):
        return torch.tensor(1.5, requires_grad=True)

    def _fake_teacher(*args, **kwargs):
        called["teacher"] += 1
        return torch.zeros(1, 4, 2, 2)

    monkeypatch.setattr(SDTrainer, "calculate_loss", _fake_super_calc)
    trainer.sd.get_turbo_teacher_prediction = _fake_teacher
    trainer.sd._sampling_transformer = nn.Linear(1, 1)

    out = trainer.calculate_loss(
        noise_pred=torch.randn(1, 4, 2, 2, requires_grad=True),
        noise=torch.randn(1, 4, 2, 2),
        noisy_latents=torch.randn(1, 4, 2, 2),
        timesteps=torch.tensor([500.0]),
        batch=SimpleNamespace(prompt_embeds=None),
        mask_multiplier=1.0,
        prior_pred=None,
    )
    assert called["teacher"] == 0
    assert float(out.detach()) == pytest.approx(1.5)
    assert not hasattr(ZImageDiffSynthTrainer, "get_turbo_teacher_prediction")
    assert not hasattr(trainer, "_turbo_teacher_pred")
    assert not hasattr(trainer, "_turbo_teacher_mse")


def test_mode_false_predict_noise_has_no_teacher_path(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from toolkit.prompt_utils import PromptEmbeds

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=False,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())

    called = {"teacher": 0}

    def _fake_super_pred(self, *args, **kwargs):
        return torch.zeros(1, 4, 2, 2, requires_grad=True)

    def _fake_teacher(*args, **kwargs):
        called["teacher"] += 1
        return torch.zeros(1, 4, 2, 2)

    monkeypatch.setattr(SDTrainer, "predict_noise", _fake_super_pred)
    trainer.sd.get_turbo_teacher_prediction = _fake_teacher
    trainer.sd._sampling_transformer = nn.Linear(1, 1)

    embeds = PromptEmbeds(torch.randn(1, 4, 8))
    trainer.predict_noise(
        noisy_latents=torch.randn(1, 4, 2, 2),
        timesteps=torch.tensor([500.0]),
        conditional_embeds=embeds,
        batch=SimpleNamespace(prompt_embeds=embeds),
        is_primary_pred=True,
    )
    assert called["teacher"] == 0


def _mark_dit_pair():
    """Paired base/sampling DiTs that record which forward ran."""

    class _MarkDit(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.w = nn.Parameter(torch.ones(1), requires_grad=True)
            self.in_channels = 16

        def forward(self, latent_list, timestep, text_embeds, return_dict=False):
            b = len(latent_list)
            h, w = latent_list[0].shape[-2], latent_list[0].shape[-1]
            c = latent_list[0].shape[0]
            scale = 2.0 if self.name == "sampling" else 1.0
            outs = [torch.ones(c, 1, h, w) * scale for _ in range(b)]
            return (outs,)

    calls = []
    base = _MarkDit("base")
    sampling = _MarkDit("sampling")
    orig_base_fwd = base.forward
    orig_samp_fwd = sampling.forward

    def _wrap(name, orig):
        def _fn(*args, **kwargs):
            calls.append(name)
            return orig(*args, **kwargs)

        return _fn

    base.forward = _wrap("base", orig_base_fwd)
    sampling.forward = _wrap("sampling", orig_samp_fwd)
    return base, sampling, calls


def _stub_model_for_noise_pred(*, train_on_turbo: bool, base, sampling, calls):
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.train_torch_dtype = torch.float32
    model.model_config = SimpleNamespace(low_vram=False)
    model.gradient_checkpointing = False
    model._train_on_turbo = train_on_turbo
    model._raw_dit = base
    model._main_is_diffusers = True
    model._sampling_is_diffusers = True
    model._sampling_transformer = SimpleNamespace(_inner_dit=sampling)
    model._sampling_network = None
    model._saved_train_network = None
    model.network = None
    model._place_training_dit = lambda device: calls.append(("place_main", str(device)))
    model._move_sampling_transformer = lambda device: calls.append(
        ("move_sampling", str(device))
    )
    model._force_network_to = lambda net, device: None
    model._flush_cuda = lambda: None
    return model


# --- Mode false: base DiT forward only (no Turbo train path) ---


def test_mode_false_get_noise_prediction_uses_base_dit():
    """_train_on_turbo=False must forward base DiT; never sampling Turbo."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )
    from toolkit.prompt_utils import PromptEmbeds

    base, sampling, calls = _mark_dit_pair()
    model = _stub_model_for_noise_pred(
        train_on_turbo=False, base=base, sampling=sampling, calls=calls
    )
    latents = torch.randn(1, 16, 4, 4)
    embeds = PromptEmbeds(torch.randn(1, 8, 16))
    pred = ZImageDiffSynthModel.get_noise_prediction(
        model,
        latents,
        torch.tensor([500.0]),
        embeds,
    )
    assert "base" in calls
    assert "sampling" not in calls
    assert any(
        isinstance(c, tuple) and c[0] == "move_sampling" and c[1] == "cpu"
        for c in calls
    )
    assert any(isinstance(c, tuple) and c[0] == "place_main" for c in calls)
    assert torch.allclose(pred, -torch.full_like(pred, 1.0))


# --- Mode true: get_noise_prediction uses sampling DiT ---


def test_mode_true_get_noise_prediction_uses_sampling_dit():
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )
    from toolkit.prompt_utils import PromptEmbeds

    base, sampling, calls = _mark_dit_pair()
    model = _stub_model_for_noise_pred(
        train_on_turbo=True, base=base, sampling=sampling, calls=calls
    )

    latents = torch.randn(1, 16, 4, 4)
    embeds = PromptEmbeds(torch.randn(1, 8, 16))
    pred = ZImageDiffSynthModel.get_noise_prediction(
        model,
        latents,
        torch.tensor([500.0]),
        embeds,
    )
    assert "sampling" in calls
    assert "base" not in calls
    # May park base on CPU; must never place base on CUDA.
    assert not any(
        isinstance(c, tuple) and c[0] == "place_main" and "cuda" in c[1]
        for c in calls
    )
    assert any(
        isinstance(c, tuple) and c[0] == "move_sampling" for c in calls
    )
    assert torch.allclose(pred, -torch.full_like(pred, 2.0))


def test_mode_true_calculate_loss_has_no_mse_teacher(monkeypatch):
    """Train-on-Turbo must not reintroduce the old MSE teacher loss path."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=True,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())

    called = {"teacher": 0}

    def _fake_super_calc(self, *args, **kwargs):
        return torch.tensor(1.5, requires_grad=True)

    def _fake_teacher(*args, **kwargs):
        called["teacher"] += 1
        return torch.zeros(1, 4, 2, 2)

    monkeypatch.setattr(SDTrainer, "calculate_loss", _fake_super_calc)
    trainer.sd.get_turbo_teacher_prediction = _fake_teacher
    trainer.sd._sampling_transformer = nn.Linear(1, 1)

    out = trainer.calculate_loss(
        noise_pred=torch.randn(1, 4, 2, 2, requires_grad=True),
        noise=torch.randn(1, 4, 2, 2),
        noisy_latents=torch.randn(1, 4, 2, 2),
        timesteps=torch.tensor([500.0]),
        batch=SimpleNamespace(prompt_embeds=None),
        mask_multiplier=1.0,
        prior_pred=None,
    )
    assert called["teacher"] == 0
    assert float(out.detach()) == pytest.approx(1.5)
    assert not hasattr(ZImageDiffSynthTrainer, "get_turbo_teacher_prediction")
    assert not hasattr(trainer, "_turbo_teacher_pred")
    assert not hasattr(trainer, "_turbo_teacher_mse")


def test_get_noise_prediction_turbo_does_not_place_main_on_cuda(monkeypatch):
    """With _train_on_turbo, get_noise_prediction must not _place_training_dit(CUDA)."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )
    from toolkit.prompt_utils import PromptEmbeds

    class _StubDit(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = 16
            self.w = nn.Parameter(torch.ones(1), requires_grad=False)

        def forward(self, latent_list, timestep, text_embeds, return_dict=False):
            outs = [
                torch.zeros(t.shape[0], 1, t.shape[-2], t.shape[-1])
                for t in latent_list
            ]
            return (outs,)

    place_calls = []
    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = torch.device("cuda:0")
    model.torch_dtype = torch.float32
    model.train_torch_dtype = torch.float32
    model.model_config = SimpleNamespace(low_vram=False)
    model.gradient_checkpointing = False
    model._train_on_turbo = True
    sampling = _StubDit()
    model._raw_dit = _StubDit()
    model._main_is_diffusers = True
    model._sampling_is_diffusers = True
    model._sampling_transformer = SimpleNamespace(_inner_dit=sampling)
    model._sampling_network = None
    model._saved_train_network = None
    model.network = None

    def _place(device):
        place_calls.append(str(device))

    model._place_training_dit = _place
    model._move_sampling_transformer = lambda device: None
    model._force_network_to = lambda net, device: None
    model._flush_cuda = lambda: None

    latents = torch.randn(1, 16, 4, 4)
    embeds = PromptEmbeds(torch.randn(1, 8, 16))
    ZImageDiffSynthModel.get_noise_prediction(
        model,
        latents,
        torch.tensor([500.0]),
        embeds,
    )
    assert all("cuda" not in d for d in place_calls)
    assert any(d == "cpu" for d in place_calls)


# --- Residency swap order ---


def test_apply_turbo_teacher_mode_residency_order(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    order = []

    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = torch.device("cuda:0")
    model.torch_dtype = torch.float32
    model._train_on_turbo = False
    model._saved_train_network = None
    model._sampling_transformer = object()
    model._raw_dit = object()
    model.model = None

    main_net = MagicMock()
    main_net.force_to = MagicMock(
        side_effect=lambda d, dt: order.append(("main_force_to", str(d)))
    )
    samp_net = MagicMock()
    samp_net.force_to = MagicMock(
        side_effect=lambda d, dt: order.append(("samp_force_to", str(d)))
    )
    p = nn.Parameter(torch.ones(1), requires_grad=True)
    main_net.parameters = MagicMock(side_effect=lambda: iter([p]))
    samp_net.parameters = MagicMock(side_effect=lambda: iter([p]))

    model.network = main_net
    model._sampling_network = samp_net

    def _place(device):
        order.append(("place_main", str(device)))

    def _move_sampling(device):
        order.append(("move_sampling", str(device)))

    def _move_main(device):
        order.append(("move_main", str(device)))

    def _flush():
        order.append(("flush",))

    model._place_training_dit = _place
    model._move_sampling_transformer = _move_sampling
    model._move_main_network = _move_main
    model._flush_cuda = _flush
    model.gradient_checkpointing = False

    # Bypass unwrap_model identity
    monkeypatch.setattr(
        "extensions_built_in.diffusion_models.z_image_diffsynth.model.unwrap_model",
        lambda x: x,
    )

    ZImageDiffSynthModel.apply_turbo_teacher_mode(model, True)
    assert model._train_on_turbo is True
    assert model.network is samp_net
    assert model._saved_train_network is main_net

    true_order = list(order)
    place_idx = true_order.index(("place_main", "cpu"))
    flush_before_samp = next(
        i
        for i, c in enumerate(true_order)
        if c == ("flush",) and i > place_idx
    )
    samp_idx = next(
        i for i, c in enumerate(true_order) if c[0] == "move_sampling" and "cuda" in c[1]
    )
    assert place_idx < flush_before_samp < samp_idx
    # Shared LoRA stays on CUDA; parking the base DiT must not force_to(CPU) on it.
    assert ("main_force_to", "cpu") not in true_order
    assert any(c[0] == "samp_force_to" and "cuda" in c[1] for c in true_order)

    order.clear()
    ZImageDiffSynthModel.apply_turbo_teacher_mode(model, False)
    assert model._train_on_turbo is False
    assert model.network is main_net
    assert model._saved_train_network is None

    false_order = list(order)
    samp_cpu_idx = next(
        i for i, c in enumerate(false_order) if c[0] == "move_sampling" and c[1] == "cpu"
    )
    flush_before_main = next(
        i
        for i, c in enumerate(false_order)
        if c == ("flush",) and i > samp_cpu_idx
    )
    move_main_idx = next(i for i, c in enumerate(false_order) if c[0] == "move_main")
    assert samp_cpu_idx < flush_before_main < move_main_idx
