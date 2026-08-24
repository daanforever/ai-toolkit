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
        network=object(),
    )

    ZImageDiffSynthTrainer.apply_runtime_turbo_teacher_mode(trainer, True)
    trainer.sd.apply_turbo_teacher_mode.assert_called_with(True)
    assert trainer.train_config.turbo_teacher_weight is True
    assert trainer.network is trainer.sd.network

    ZImageDiffSynthTrainer.apply_runtime_turbo_teacher_mode(trainer, False)
    trainer.sd.apply_turbo_teacher_mode.assert_called_with(False)
    assert trainer.train_config.turbo_teacher_weight is False
    assert trainer.network is trainer.sd.network


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
        network=object(),
    )
    monkeypatch.setattr(
        trainer, "get_runtime_turbo_teacher_weight", lambda: True
    )

    DiffusionTrainer.apply_runtime_turbo_teacher_weight(trainer)
    trainer.sd.apply_turbo_teacher_mode.assert_called_with(True)
    assert trainer.train_config.turbo_teacher_weight is True
    assert trainer._last_applied_runtime_turbo_teacher_weight is True
    assert trainer.network is trainer.sd.network


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


# --- Turbo LoRA is_active desync (P1 red test; passes after P2 trainer.network sync) ---


def _tiny_shared_lora_pair():
    """Two tiny UNets + LoRASpecialNetworks; sampling shares params with base (CPU)."""
    from toolkit.lora_special import LoRASpecialNetwork

    class DummyTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()

    class LoRACompatibleLinear(nn.Linear):
        pass

    class UNet2DConditionModel(nn.Module):  # type: ignore[override]
        def __init__(self):
            super().__init__()
            self.block = nn.Module()
            self.block.linear = LoRACompatibleLinear(4, 4)

        def forward(self, x):
            return self.block.linear(x)

    text_enc = DummyTextEncoder()
    mod_a = UNet2DConditionModel()
    mod_b = UNet2DConditionModel()
    common = dict(
        text_encoder=text_enc,
        train_text_encoder=False,
        train_unet=True,
        lora_dim=2,
        alpha=1.0,
        target_lin_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE,
        target_conv_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3,
    )
    base_net = LoRASpecialNetwork(unet=mod_a, **common)
    sampling_net = LoRASpecialNetwork(unet=mod_b, **common)
    assert base_net.unet_loras, "expected base unet LoRA modules"
    assert sampling_net.unet_loras, "expected sampling unet LoRA modules"
    sampling_net.share_parameters_with(base_net)
    sampling_net._update_torch_multiplier()
    base_net._update_torch_multiplier()
    base_net.apply_to(text_enc, mod_a, False, True)
    sampling_net.apply_to(text_enc, mod_b, False, True)
    return mod_a, mod_b, base_net, sampling_net


def _stub_sd_for_turbo_lora(mod_b, base_net, sampling_net):
    """Real apply_turbo_teacher_mode on __new__ instance; residency moves are no-ops."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    sd = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    sd.device_torch = torch.device("cpu")
    sd.torch_dtype = torch.float32
    sd._train_on_turbo = False
    sd._saved_train_network = None
    sd._raw_dit = object()
    sd.model = None
    sd.gradient_checkpointing = False
    sd.network = base_net
    sd._sampling_network = sampling_net
    sd._sampling_transformer = mod_b
    sd._place_training_dit = lambda device: None
    sd._move_sampling_transformer = lambda device: None
    sd._move_main_network = lambda device: None
    sd._flush_cuda = lambda: None
    sd._force_network_to = lambda net, device: None
    return sd


def test_turbo_lora_is_active_desync_with_base_context():
    """with base_net: sampling hooks stay inactive → shared LoRA .grad is None."""
    _mod_a, mod_b, base_net, sampling_net = _tiny_shared_lora_pair()
    x = torch.randn(1, 4)
    with base_net:
        assert base_net.is_active
        assert not sampling_net.is_active
        loss = mod_b(x).sum()
        loss.backward()
    for p in sampling_net.parameters():
        if p.requires_grad:
            assert p.grad is None


def test_trainer_network_not_synced_after_turbo_teacher_mode(monkeypatch):
    """
    After apply_turbo_teacher_mode, trainer.network must track sd.network (sampling LoRA)
    so ``with self.network`` activates Turbo hooks and LoRA grads/SGD step succeed.
    """
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    _mod_a, mod_b, base_net, sampling_net = _tiny_shared_lora_pair()
    sd = _stub_sd_for_turbo_lora(mod_b, base_net, sampling_net)

    monkeypatch.setattr(
        "extensions_built_in.diffusion_models.z_image_diffsynth.model.unwrap_model",
        lambda x: x,
    )
    monkeypatch.setattr(SDTrainer, "hook_before_train_loop", lambda self: None)
    monkeypatch.setattr(DiffusionTrainer, "hook_before_train_loop", lambda self: None)

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=True,
        sampling_name_or_path="/tmp/turbo",
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())
    trainer.is_ui_trainer = False
    trainer.network = base_net
    trainer.sd = sd

    # Do NOT assign trainer.network = sd.network here (that is P2).
    trainer.hook_before_train_loop()
    assert trainer.sd.network is sampling_net
    assert trainer.network is trainer.sd.network

    trainer.apply_runtime_turbo_teacher_mode(True)
    assert trainer.network is trainer.sd.network

    shared = [p for p in sampling_net.parameters() if p.requires_grad]
    assert shared
    for p in shared:
        if p.grad is not None:
            p.grad = None
    before = [p.detach().clone() for p in shared]
    opt = torch.optim.SGD(shared, lr=1.0)
    x = torch.randn(1, 4)
    with trainer.network:
        loss = mod_b(x).sum()
        loss.backward()
    assert all(p.grad is not None for p in shared)
    opt.step()
    assert any(not torch.equal(p, b) for p, b in zip(shared, before))


# --- Unload sampling transformer after generate ---


def _stub_base_for_unload(*, train_on_turbo=None):
    from toolkit.models.base_model import BaseModel

    to_devices = []
    model = BaseModel.__new__(BaseModel)
    model.torch_dtype = torch.float32
    sampling = MagicMock()

    def _to(device, *args, **kwargs):
        to_devices.append(device)
        return sampling

    sampling.to = _to
    model._sampling_transformer = sampling
    if train_on_turbo is not None:
        model._train_on_turbo = train_on_turbo
    return model, to_devices


def test_unload_sampling_transformer_after_generate_respects_train_on_turbo(
    monkeypatch,
):
    prints = []
    monkeypatch.setattr(
        "toolkit.models.base_model.print_acc",
        lambda *a, **k: prints.append(a[0] if a else ""),
    )
    model, to_devices = _stub_base_for_unload(train_on_turbo=True)

    model._unload_sampling_transformer_after_generate()
    assert to_devices == []
    assert prints == []

    model._train_on_turbo = False
    model._unload_sampling_transformer_after_generate()
    assert to_devices == ["cpu"]
    assert any("Unloaded sampling transformer to CPU" in p for p in prints)


def test_unload_sampling_transformer_when_train_on_turbo_unset(monkeypatch):
    prints = []
    monkeypatch.setattr(
        "toolkit.models.base_model.print_acc",
        lambda *a, **k: prints.append(a[0] if a else ""),
    )
    model, to_devices = _stub_base_for_unload()
    assert not hasattr(model, "_train_on_turbo")

    model._unload_sampling_transformer_after_generate()
    assert to_devices == ["cpu"]
    assert any("Unloaded sampling transformer to CPU" in p for p in prints)


# --- Sampling transformer .to residency (spy real move, not debug label alone) ---


@pytest.fixture
def _debug_enabled_restored():
    """Enable process-global debug; restore prior config after the test."""
    import toolkit.util.debug as debug_mod

    prev = debug_mod._debug_config
    debug_mod.set_debug_config(SimpleNamespace(debug=True))
    try:
        yield
    finally:
        debug_mod.set_debug_config(prev)


def _frozen_sampling_inner():
    """Frozen non-LoRA Linear so _first_frozen_base_param can detect need_move."""

    class _InnerDit(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen_linear = nn.Linear(4, 4, bias=False)
            self.frozen_linear.weight.requires_grad_(False)
            self.lora_linear = nn.Linear(4, 4, bias=False)
            self.lora_linear.weight.requires_grad_(True)

    return _InnerDit()


def _spy_wrapper_to(wrapper):
    """Instance-bind spy on _DiTUnetWrapper.to; returns (to_calls list, restore)."""
    to_calls = []
    orig_to = wrapper.to

    def _spy(*args, **kwargs):
        to_calls.append(args[0] if args else kwargs.get("device"))
        return orig_to(*args, **kwargs)

    wrapper.to = _spy
    return to_calls, lambda: setattr(wrapper, "to", orig_to)


def _model_for_move_sampling(wrapper, statuses):
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.print_and_status_update = lambda msg: statuses.append(str(msg))
    model._sampling_transformer = wrapper
    model.device_torch = torch.device("cpu")
    return model


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for cpu→cuda sampling move",
)
def test_move_sampling_transformer_to_on_device_change_then_noop(_debug_enabled_restored):
    """(a) first different-device move calls .to + status; (b) same-device repeat skips both."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
        _DiTUnetWrapper,
    )

    inner = _frozen_sampling_inner()
    inner.frozen_linear.to("cpu")
    wrapper = _DiTUnetWrapper(inner)
    statuses = []
    model = _model_for_move_sampling(wrapper, statuses)
    to_calls, restore_to = _spy_wrapper_to(wrapper)

    try:
        statuses.clear()
        to_calls.clear()
        ZImageDiffSynthModel._move_sampling_transformer(model, "cuda")
        assert len(to_calls) == 1
        assert any("moving sampling transformer" in s for s in statuses)

        statuses.clear()
        to_calls.clear()
        ZImageDiffSynthModel._move_sampling_transformer(model, "cuda")
        assert to_calls == []
        assert not any("moving sampling transformer" in s for s in statuses)
    finally:
        restore_to()
        try:
            ZImageDiffSynthModel._move_sampling_transformer(model, "cpu")
        except Exception:
            pass


def test_move_sampling_transformer_noop_no_debug_label(_debug_enabled_restored):
    """Post-fix: same-device no-op must not emit memory_debug [DEBUG Move ...] label."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
        _DiTUnetWrapper,
    )

    inner = _frozen_sampling_inner()
    wrapper = _DiTUnetWrapper(inner)
    statuses = []
    model = _model_for_move_sampling(wrapper, statuses)
    to_calls, restore_to = _spy_wrapper_to(wrapper)

    try:
        statuses.clear()
        to_calls.clear()
        ZImageDiffSynthModel._move_sampling_transformer(model, "cpu")
        assert to_calls == []
        assert not any("[DEBUG Move sampling transformer]" in s for s in statuses)
    finally:
        restore_to()


def test_get_noise_prediction_repeats_do_not_call_sampling_to():
    """Pinned residency: three True + three False get_noise_prediction → sampling .to stays 0."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
        _DiTUnetWrapper,
    )
    from toolkit.prompt_utils import PromptEmbeds

    class _MarkDit(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.w = nn.Parameter(torch.ones(1), requires_grad=False)
            self.in_channels = 16

        def forward(self, latent_list, timestep, text_embeds, return_dict=False):
            b = len(latent_list)
            h, w = latent_list[0].shape[-2], latent_list[0].shape[-1]
            c = latent_list[0].shape[0]
            scale = 2.0 if self.name == "sampling" else 1.0
            outs = [torch.ones(c, 1, h, w) * scale for _ in range(b)]
            return (outs,)

    base = _MarkDit("base")
    sampling = _MarkDit("sampling")
    wrapper = _DiTUnetWrapper(sampling)

    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.train_torch_dtype = torch.float32
    model.model_config = SimpleNamespace(low_vram=False)
    model.gradient_checkpointing = False
    model._raw_dit = base
    model._main_is_diffusers = True
    model._sampling_is_diffusers = True
    model._sampling_transformer = wrapper
    model._sampling_network = None
    model._saved_train_network = None
    model.network = None
    model.print_and_status_update = lambda *a, **k: None
    model._flush_cuda = lambda: None
    model._force_network_to = lambda net, device: None

    base.to("cpu")
    _DiTUnetWrapper.to(wrapper, "cpu")

    to_calls, restore_to = _spy_wrapper_to(wrapper)
    latents = torch.randn(1, 16, 4, 4)
    embeds = PromptEmbeds(torch.randn(1, 8, 16))
    timesteps = torch.tensor([500.0])

    try:
        model._train_on_turbo = True
        for _ in range(3):
            ZImageDiffSynthModel.get_noise_prediction(
                model, latents, timesteps, embeds
            )
        assert to_calls == []

        model._train_on_turbo = False
        for _ in range(3):
            ZImageDiffSynthModel.get_noise_prediction(
                model, latents, timesteps, embeds
            )
        assert to_calls == []
    finally:
        restore_to()


def _mark_dit(name):
    """Tiny frozen DiT for get_noise_prediction residency/flush tests."""

    class _MarkDit(nn.Module):
        def __init__(self):
            super().__init__()
            self.name = name
            self.w = nn.Parameter(torch.ones(1), requires_grad=False)
            self.in_channels = 16

        def forward(self, latent_list, timestep, text_embeds, return_dict=False):
            b = len(latent_list)
            h, w = latent_list[0].shape[-2], latent_list[0].shape[-1]
            c = latent_list[0].shape[0]
            scale = 2.0 if self.name == "sampling" else 1.0
            outs = [torch.ones(c, 1, h, w) * scale for _ in range(b)]
            return (outs,)

    return _MarkDit()


def _model_for_flush_residency(base, wrapper):
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = torch.device("cuda")
    model.torch_dtype = torch.float32
    model.train_torch_dtype = torch.float32
    model.model_config = SimpleNamespace(low_vram=False)
    model.gradient_checkpointing = False
    model._raw_dit = base
    model.model = None
    model._main_is_diffusers = True
    model._sampling_is_diffusers = True
    model._sampling_transformer = wrapper
    model._sampling_network = None
    model._saved_train_network = None
    model.network = None
    model.print_and_status_update = lambda *a, **k: None
    model._force_network_to = lambda net, device: None
    return model


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for flush gating residency tests",
)
def test_get_noise_prediction_repeat_does_not_flush_when_already_parked():
    """Pinned residency: inactive DiT already on CPU → park is no-op → no _flush_cuda."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
        _DiTUnetWrapper,
    )
    from toolkit.prompt_utils import PromptEmbeds

    base = _mark_dit("base")
    sampling = _mark_dit("sampling")
    wrapper = _DiTUnetWrapper(sampling)
    model = _model_for_flush_residency(base, wrapper)

    flush_count = {"n": 0}
    orig_flush = model._flush_cuda

    def _spy_flush():
        flush_count["n"] += 1
        # no-op body is enough for counting; keep call site reachable

    model._flush_cuda = _spy_flush

    latents = torch.randn(1, 16, 4, 4)
    embeds = PromptEmbeds(torch.randn(1, 8, 16))
    timesteps = torch.tensor([500.0])

    try:
        # Turbo: base already on CPU, sampling on CUDA
        base.to("cpu")
        wrapper.to("cuda")
        model._train_on_turbo = True
        flush_count["n"] = 0
        for _ in range(2):
            ZImageDiffSynthModel.get_noise_prediction(
                model, latents, timesteps, embeds
            )
        assert flush_count["n"] == 0

        # Base train: sampling already on CPU, base on CUDA
        wrapper.to("cpu")
        base.to("cuda")
        model._train_on_turbo = False
        flush_count["n"] = 0
        for _ in range(2):
            ZImageDiffSynthModel.get_noise_prediction(
                model, latents, timesteps, embeds
            )
        assert flush_count["n"] == 0
    finally:
        model._flush_cuda = orig_flush
        try:
            base.to("cpu")
            wrapper.to("cpu")
        except Exception:
            pass


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for flush gating residency tests",
)
def test_get_noise_prediction_first_park_off_cuda_flushes_then_noop():
    """First park that moves inactive DiT off CUDA flushes; repeat does not."""
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
        _DiTUnetWrapper,
    )
    from toolkit.prompt_utils import PromptEmbeds

    base = _mark_dit("base")
    sampling = _mark_dit("sampling")
    wrapper = _DiTUnetWrapper(sampling)
    model = _model_for_flush_residency(base, wrapper)

    flush_count = {"n": 0}

    def _spy_flush():
        flush_count["n"] += 1

    model._flush_cuda = _spy_flush

    latents = torch.randn(1, 16, 4, 4)
    embeds = PromptEmbeds(torch.randn(1, 8, 16))
    timesteps = torch.tensor([500.0])

    try:
        base.to("cuda")
        wrapper.to("cpu")
        model._train_on_turbo = True

        ZImageDiffSynthModel.get_noise_prediction(model, latents, timesteps, embeds)
        assert flush_count["n"] >= 1
        after_first = flush_count["n"]

        ZImageDiffSynthModel.get_noise_prediction(model, latents, timesteps, embeds)
        assert flush_count["n"] == after_first
    finally:
        try:
            base.to("cpu")
            wrapper.to("cpu")
        except Exception:
            pass
