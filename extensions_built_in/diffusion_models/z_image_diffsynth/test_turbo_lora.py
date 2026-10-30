"""
CPU unit tests for train.turbo_lora.

No real model weights. Run from the repo root:
  venv\\Scripts\\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_turbo_lora.py -q
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from extensions_built_in.diffusion_models.z_image_diffsynth.turbo_lora import (
    format_turbo_lora_weight,
    run_imitation_pass,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.turbo_schedule import (
    get_turbo_sigmas_and_timesteps,
)
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer, batch_num_samples
from toolkit.config_modules import TrainConfig
from toolkit.optimizers.adafactor import Adafactor
from toolkit.prompt_utils import PromptEmbeds
from toolkit.timer import Timer

_CENTERS_8 = [1000, 954.545, 900, 833.333, 750, 642.857, 500, 300]
_OPTIMIZER_ERROR = "turbo_lora=true requires train.optimizer='adafactor' (got {opt!r})."
_SAMPLING_ERROR = (
    "turbo_lora=true requires model.sampling_name_or_path "
    "(Z-Image-Turbo / _sampling_transformer)."
)
_TEACHER_ERROR = "turbo_lora=true is incompatible with train.turbo_teacher_weight=true."
_COMPILE_UNSUPPORTED = (
    "zimage_diffsynth does not support model.compile; set model.compile: false."
)
_CACHE_ERROR = "turbo_lora=true requires train.cache_text_embeddings=true."
_TRANSFORMER_ERROR = (
    "turbo_lora=true requires a loaded sampling transformer "
    "(set model.sampling_name_or_path to Z-Image-Turbo)."
)
_NETWORK_ERROR = (
    "turbo_lora=true requires a loaded sampling network "
    "(shared LoRA on the sampling DiT)."
)
_EMBEDS_ERROR = "turbo_lora=true requires cached prompt embeds on the batch."


class _LoRAModule:
    def __init__(self, rank_dropout, module_dropout):
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout


class _Net:
    def __init__(self, rank_dropout=0.4, module_dropout=0.2):
        self.is_active = False
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.lora = _LoRAModule(rank_dropout, module_dropout)

    def __enter__(self):
        self.is_active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.is_active = False
        return False

    def get_all_modules(self):
        return [self.lora]


class _Residency:
    def __init__(self, events):
        self.base = "cuda"
        self.sampling = "cpu"
        self.events = events

    def park_base(self):
        moved = self.base != "cpu"
        self.base = "cpu"
        if moved:
            self.events.append("park_base")
        return moved

    def park_sampling(self):
        moved = self.sampling != "cpu"
        self.sampling = "cpu"
        if moved:
            self.events.append("park_turbo")
        return moved

    def host_turbo(self, device):
        target = "cpu" if _is_cpu(device) else "cuda"
        moved = self.sampling != target
        self.sampling = target
        if moved and target == "cuda":
            self.events.append("host_turbo")
        return moved

    def host_base(self, device):
        target = "cpu" if _is_cpu(device) else "cuda"
        moved = self.base != target
        self.base = target
        if moved and target == "cuda":
            self.events.append("host_base")


def _is_cpu(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cpu"
    text = str(device)
    return text == "cpu" or text.startswith("cpu")


def _batch(n, embeds=None, device=None):
    latents = torch.zeros(n, 16, 2, 2, device=device)
    if embeds is None:
        embeds = PromptEmbeds(torch.randn(n, 4, 8))
    return SimpleNamespace(latents=latents, prompt_embeds=embeds)


def _make_dit(name, shared, base_net, samp_net, events, residency, snapshots, fail_on):
    class _Dit(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = 16

        def forward(self, latent_list, timestep, text_embeds, return_dict=False):
            if name == "sampling" and samp_net.is_active and fail_on[0]:
                raise RuntimeError("turbo_on failed")
            active_base = base_net.is_active
            active_samp = samp_net.is_active
            snapshots.append(
                {
                    "dit": name,
                    "flag": residency.sd._train_on_turbo,
                    "base_active": active_base,
                    "samp_active": active_samp,
                    "base_device": residency.base,
                    "sampling_device": residency.sampling,
                    "grad_enabled": torch.is_grad_enabled(),
                    "rank_dropout": base_net.lora.rank_dropout,
                    "module_dropout": samp_net.lora.module_dropout,
                }
            )
            if name == "base" and not active_base:
                events.append("base_off")
            elif name == "base":
                events.append("base_on")
            elif not active_samp:
                events.append("turbo_off")
            ref = latent_list[0]
            piece = torch.zeros(
                ref.shape[0], 1, ref.shape[-2], ref.shape[-1], device=ref.device
            )
            if name == "base":
                piece = piece * shared.float()
            elif active_samp:
                fail_on[1] += 1
                piece = piece - (shared.float() * fail_on[1])
            return ([piece for _ in latent_list],)

    return _Dit()


def _build_world(shared, *, fail_turbo_on=False, weight=2.0, slots=8):
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    events = []
    snapshots = []
    force_devices = []
    add_calls = {"n": 0}
    fail_on = [fail_turbo_on, 0]
    residency = _Residency(events)
    base_net = _Net()
    samp_net = _Net()
    base_dit = _make_dit(
        "base", shared, base_net, samp_net, events, residency, snapshots, fail_on
    )
    samp_dit = _make_dit(
        "sampling", shared, base_net, samp_net, events, residency, snapshots, fail_on
    )
    sd = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    residency.sd = sd
    sd.device_torch = torch.device("cuda")
    sd.is_multistage = False
    sd.torch_dtype = torch.float32
    sd.train_torch_dtype = torch.float32
    sd.model_config = SimpleNamespace(low_vram=False)
    sd.gradient_checkpointing = False
    sd._train_on_turbo = False
    sd._raw_dit = base_dit
    sd._main_is_diffusers = True
    sd._sampling_is_diffusers = True
    sd._sampling_transformer = SimpleNamespace(_inner_dit=samp_dit)
    sd._sampling_network = samp_net
    sd.network = base_net
    sd._saved_train_network = None
    sd._park_base_dit_and_lora = residency.park_base
    sd._park_sampling_dit_and_lora = residency.park_sampling
    sd._move_sampling_transformer = residency.host_turbo
    sd._move_main_network = residency.host_base
    sd._flush_cuda = lambda: events.append("flush")
    sd.apply_turbo_teacher_mode = MagicMock()

    def _force(net, device):
        kind = device.type if isinstance(device, torch.device) else _is_cpu(device)
        force_devices.append("cpu" if kind is True or kind == "cpu" else "cuda")

    sd._force_network_to = _force

    def _add_noise(original, noise, timesteps):
        add_calls["n"] += 1
        step = (timesteps.to(device=original.device, dtype=original.dtype) / 1000).view(
            -1, 1, 1, 1
        )
        return (1.0 - step) * original + step * noise

    sd.add_noise = _add_noise

    def _backward(loss):
        events.append("turbo_on_backward")
        loss.backward()

    trainer = SimpleNamespace(
        sd=sd,
        network=base_net,
        train_config=SimpleNamespace(
            turbo_lora=True,
            turbo_lora_weight=weight,
            turbo_prior_steps=slots,
            content_or_style="style",
            optimizer="adafactor",
        ),
        accelerator=SimpleNamespace(backward=_backward),
    )
    return SimpleNamespace(
        trainer=trainer,
        sd=sd,
        shared=shared,
        base_net=base_net,
        samp_net=samp_net,
        events=events,
        snapshots=snapshots,
        force_devices=force_devices,
        add_calls=add_calls,
        residency=residency,
        fail_on=fail_on,
    )


def _assert_dropout_restored(world):
    for net in (world.base_net, world.samp_net):
        assert net.rank_dropout == 0.4
        assert net.module_dropout == 0.2
        assert net.lora.rank_dropout == 0.4
        assert net.lora.module_dropout == 0.2
        assert net.is_active is False


def _bind_loop(world, optimizer, concept, *, accumulate=False):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    trainer = ZImageDiffSynthTrainer.__new__(ZImageDiffSynthTrainer)
    trainer.sd = world.sd
    trainer.network = world.base_net
    trainer.train_config = world.trainer.train_config
    trainer.optimizer = optimizer
    trainer.train_single_accumulation = concept
    trainer.steps_this_boundary = 0
    trainer.is_grad_accumulation_step = accumulate
    trainer.model_config = SimpleNamespace(low_vram=False)
    trainer.embedding = None
    trainer.adapter = None
    trainer.ema = None
    trainer.lr_scheduler = SimpleNamespace(step=lambda: None)
    trainer.timer = Timer()
    trainer.network_config = SimpleNamespace(network_kwargs={"stable_lora_shrink": 0})
    trainer.accelerator = world.trainer.accelerator
    return trainer


def _install_trainer_init(monkeypatch, **overrides):
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

    def _init(self, process_id, job, config, **kwargs):
        model = (config or {}).get("model", {}) or {}
        self.config = config
        self.progress_bar = None
        self.print = lambda *a, **k: None
        self.model_config = SimpleNamespace(
            model_kwargs=dict(model.get("model_kwargs") or {}),
            sampling_name_or_path=overrides.get(
                "sampling_name_or_path", model.get("sampling_name_or_path")
            ),
            compile=overrides.get("compile", False),
            low_vram=False,
        )
        self.train_config = SimpleNamespace(
            noise_scheduler="placeholder",
            num_train_timesteps=None,
            loss_type=None,
            timestep_type=overrides.get("timestep_type", "turbo_prior"),
            content_or_style="balanced",
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
            turbo_teacher_weight=overrides.get("turbo_teacher_weight", False),
            turbo_lora=overrides.get("turbo_lora", False),
            turbo_lora_weight=overrides.get("turbo_lora_weight", 1.0),
            optimizer=overrides.get("optimizer", "adafactor"),
            cache_text_embeddings=overrides.get("cache_text_embeddings", True),
            pred_scaler=1.0,
            match_noise_norm=False,
        )
        self.logger = MagicMock()
        self.accelerator = SimpleNamespace(is_main_process=True)
        self.sd = SimpleNamespace()
        self.is_ui_trainer = False
        self.is_caching_text_embeddings = False

    monkeypatch.setattr(DiffusionTrainer, "__init__", _init)


def _base_cfg():
    return {
        "model": {
            "sampling_name_or_path": "/tmp/turbo",
            "model_kwargs": {"use_diffsynth_training_loop": False},
        }
    }


def test_documented_eight_centers():
    _, centers = get_turbo_sigmas_and_timesteps(
        num_inference_steps=8,
        use_dynamic_shifting=False,
    )
    assert torch.allclose(centers.float(), torch.tensor(_CENTERS_8), atol=0.01)


def test_format_turbo_lora_weight_strips_trailing_zeros():
    assert format_turbo_lora_weight(1.0) == "1"
    assert format_turbo_lora_weight(1.5) == "1.5"
    assert format_turbo_lora_weight(0.5) == "0.5"
    assert format_turbo_lora_weight(2) == "2"


def test_parent_after_concept_hook_is_noop():
    assert SDTrainer._after_train_single_accumulation(SimpleNamespace(), None, 1.0) is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"turbo_lora": "yes"}, "turbo_lora must be boolean true/false; got 'yes'"),
        ({"turbo_lora": 1}, "turbo_lora must be boolean true/false; got 1"),
        (
            {"turbo_lora": True, "turbo_lora_weight": True},
            "turbo_lora_weight must be a finite number > 0, got True",
        ),
        (
            {"turbo_lora": True, "turbo_lora_weight": 0},
            "turbo_lora_weight must be a finite number > 0, got 0",
        ),
        (
            {"turbo_lora": True, "turbo_lora_weight": -1.5},
            "turbo_lora_weight must be a finite number > 0, got -1.5",
        ),
        (
            {"turbo_lora": True, "turbo_lora_weight": float("nan")},
            "turbo_lora_weight must be a finite number > 0, got nan",
        ),
        (
            {"turbo_lora": True, "turbo_lora_weight": float("inf")},
            "turbo_lora_weight must be a finite number > 0, got inf",
        ),
        (
            {"turbo_lora": True, "turbo_lora_weight": "1"},
            "turbo_lora_weight must be a finite number > 0, got '1'",
        ),
    ],
)
def test_train_config_rejects_bad_turbo_lora_values(kwargs, message):
    with pytest.raises(ValueError) as exc:
        TrainConfig(**kwargs)
    assert str(exc.value) == message


def test_train_config_turbo_lora_defaults_and_ignores_weight_when_false():
    missing = TrainConfig(**{})
    assert missing.turbo_lora is False
    assert missing.turbo_lora_weight == 1.0
    none_flag = TrainConfig(**{"turbo_lora": None, "turbo_lora_weight": "bad"})
    assert none_flag.turbo_lora is False
    assert none_flag.turbo_lora_weight == 1.0
    false_flag = TrainConfig(**{"turbo_lora": False, "turbo_lora_weight": 0})
    assert false_flag.turbo_lora is False
    assert false_flag.turbo_lora_weight == 1.0
    enabled = TrainConfig(**{"turbo_lora": True})
    assert enabled.turbo_lora is True
    assert enabled.turbo_lora_weight == 1.0
    weighted = TrainConfig(**{"turbo_lora": True, "turbo_lora_weight": 2})
    assert weighted.turbo_lora_weight == 2.0


def test_init_accepts_adafactor_any_case(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _install_trainer_init(monkeypatch, turbo_lora=True, optimizer="Adafactor")
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())
    assert trainer.train_config.turbo_lora is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"optimizer": "adamw"}, _OPTIMIZER_ERROR.format(opt="adamw")),
        ({"optimizer": "hfadafactor"}, _OPTIMIZER_ERROR.format(opt="hfadafactor")),
        ({"optimizer": None}, _OPTIMIZER_ERROR.format(opt=None)),
        ({"sampling_name_or_path": None}, _SAMPLING_ERROR),
        ({"sampling_name_or_path": ""}, _SAMPLING_ERROR),
        ({"turbo_teacher_weight": True}, _TEACHER_ERROR),
        ({"cache_text_embeddings": False}, _CACHE_ERROR),
        ({"cache_text_embeddings": None}, _CACHE_ERROR),
    ],
)
def test_init_turbo_lora_value_errors(monkeypatch, overrides, message):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    fields = {"turbo_lora": True, "optimizer": "adafactor", "cache_text_embeddings": True}
    fields.update(overrides)
    _install_trainer_init(monkeypatch, **fields)
    with pytest.raises(ValueError) as exc:
        ZImageDiffSynthTrainer(0, None, _base_cfg())
    assert str(exc.value) == message


@pytest.mark.parametrize("turbo_lora", [False, True])
def test_init_rejects_model_compile(monkeypatch, turbo_lora):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _install_trainer_init(
        monkeypatch,
        compile=True,
        turbo_lora=turbo_lora,
        optimizer="adafactor",
        cache_text_embeddings=True,
    )
    with pytest.raises(ValueError) as exc:
        ZImageDiffSynthTrainer(0, None, _base_cfg())
    assert str(exc.value) == _COMPILE_UNSUPPORTED


def test_init_accepts_model_compile_false(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    _install_trainer_init(monkeypatch, compile=False, turbo_lora=False)
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())
    assert trainer.model_config.compile is False


@pytest.mark.parametrize(
    ("sd_attrs", "message"),
    [
        (
            {"_sampling_transformer": None, "_sampling_network": object()},
            _TRANSFORMER_ERROR,
        ),
        (
            {"_sampling_transformer": object(), "_sampling_network": None},
            _NETWORK_ERROR,
        ),
    ],
)
def test_pass_requires_sampling_modules(sd_attrs, message):
    trainer = SimpleNamespace(
        sd=SimpleNamespace(**sd_attrs),
        train_config=SimpleNamespace(turbo_lora_weight=1.0, turbo_prior_steps=8),
        accelerator=SimpleNamespace(backward=lambda loss: None),
    )
    batch = _batch(1)
    with pytest.raises(ValueError) as exc:
        run_imitation_pass(trainer, batch, 1.0)
    assert str(exc.value) == message


def test_pass_requires_cached_prompt_embeds():
    trainer = SimpleNamespace(
        sd=SimpleNamespace(_sampling_transformer=object(), _sampling_network=object()),
        train_config=SimpleNamespace(turbo_lora_weight=1.0, turbo_prior_steps=8),
        accelerator=SimpleNamespace(backward=lambda loss: None),
    )
    batch = SimpleNamespace(latents=torch.zeros(1, 16, 2, 2), prompt_embeds=None)
    with pytest.raises(ValueError) as exc:
        run_imitation_pass(trainer, batch, 1.0)
    assert str(exc.value) == _EMBEDS_ERROR


def test_imitation_pass_contract(monkeypatch):
    import extensions_built_in.diffusion_models.z_image_diffsynth.turbo_lora.turbo_lora as turbo_lora
    import extensions_built_in.diffusion_models.z_image_diffsynth.turbo_schedule as schedule

    def _forbid_slot_weights(*args, **kwargs):
        raise AssertionError("content_or_style slot weights are not an imitation t")

    monkeypatch.setattr(schedule, "turbo_slot_sampling_weights", _forbid_slot_weights)
    monkeypatch.setattr(
        turbo_lora.torch,
        "randint",
        lambda low, high, size, **kwargs: torch.zeros(size, dtype=torch.long),
    )
    shared = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
    device_before = shared.device
    world = _build_world(shared, weight=2.0)
    seen = []
    grad_at_backward = {}

    def _backward(loss):
        grad_at_backward["value"] = None if shared.grad is None else shared.grad.detach().clone()
        grad_at_backward["loss"] = float(loss.detach())
        world.events.append("turbo_on_backward")
        loss.backward()

    world.trainer.accelerator.backward = _backward
    orig = type(world.sd).get_noise_prediction
    calls = []

    def _wrapped(self, latent_model_input, timestep, text_embeddings, batch=None, **kwargs):
        calls.append(
            {
                "latent": latent_model_input,
                "timestep": timestep,
                "embeds": text_embeddings,
                "batch": batch,
                "kwargs": dict(kwargs),
            }
        )
        pred = orig(
            self,
            latent_model_input,
            timestep,
            text_embeddings,
            batch=batch,
            **kwargs,
        )
        calls[-1]["pred"] = pred.detach().clone()
        calls[-1]["in_graph"] = pred.grad_fn is not None
        return pred

    world.sd.get_noise_prediction = _wrapped.__get__(world.sd, type(world.sd))
    batch = _batch(1)
    mse = run_imitation_pass(world.trainer, batch, 1.0)

    assert world.events == [
        "base_off",
        "base_on",
        "park_base",
        "flush",
        "host_turbo",
        "turbo_off",
        "turbo_on_backward",
        "park_turbo",
        "flush",
        "host_base",
    ]
    assert [snap["flag"] for snap in world.snapshots] == [False, False, True, True]
    base_off, base_on, turbo_off, turbo_on = world.snapshots
    assert base_off["dit"] == "base"
    assert base_off["base_active"] is False
    assert base_off["sampling_device"] == "cpu"
    assert base_off["grad_enabled"] is False
    assert base_on["dit"] == "base"
    assert base_on["base_active"] is True
    assert base_on["sampling_device"] == "cpu"
    assert base_on["grad_enabled"] is False
    assert turbo_off["dit"] == "sampling"
    assert turbo_off["samp_active"] is False
    assert turbo_off["base_device"] == "cpu"
    assert turbo_off["base_active"] is False
    assert turbo_on["dit"] == "sampling"
    assert turbo_on["samp_active"] is True
    assert turbo_on["base_active"] is False
    assert turbo_on["base_device"] == "cpu"
    assert turbo_on["grad_enabled"] is True
    assert all(snap["rank_dropout"] is None for snap in world.snapshots)
    assert all(snap["module_dropout"] is None for snap in world.snapshots)
    assert [call["in_graph"] for call in calls] == [False, False, False, True]
    base_off_pred, base_on_pred, turbo_off_pred, turbo_on_pred = (
        call["pred"] for call in calls
    )
    target = turbo_off_pred + base_on_pred - base_off_pred
    assert target.grad_fn is None
    assert torch.allclose(target, torch.zeros_like(target))
    assert torch.allclose(turbo_on_pred, torch.full_like(turbo_on_pred, 0.25))
    assert float(mse) == pytest.approx(0.0625)
    assert grad_at_backward["value"] is None
    assert grad_at_backward["loss"] == pytest.approx(0.125)
    assert shared.grad is not None
    assert float(shared.grad) == pytest.approx(1.0)
    assert shared.device == device_before
    assert all(device != "cpu" for device in world.force_devices)
    assert world.add_calls["n"] == 1
    assert len({id(call["latent"]) for call in calls}) == 1
    assert len({id(call["timestep"]) for call in calls}) == 1
    assert all(call["embeds"] is batch.prompt_embeds for call in calls)
    assert all(call["batch"] is None and call["kwargs"] == {} for call in calls)
    assert calls[0]["timestep"].shape == (1,)
    assert torch.allclose(
        calls[0]["timestep"],
        torch.full((1,), _CENTERS_8[0], dtype=calls[0]["timestep"].dtype),
        atol=0.01,
    )
    assert world.sd._train_on_turbo is False
    assert world.residency.base == "cuda"
    assert world.residency.sampling == "cpu"
    _assert_dropout_restored(world)
    world.sd.apply_turbo_teacher_mode.assert_not_called()
    assert world.trainer.network is world.base_net
    assert world.sd.network is world.base_net


def test_imitation_places_prompt_embeds_on_latent_device():
    shared = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
    world = _build_world(shared, weight=1.0)
    batch = _batch(1)
    calls = []
    real_to = batch.prompt_embeds.to

    def _to(*args, **kwargs):
        calls.append((args, kwargs))
        return real_to(*args, **kwargs)

    batch.prompt_embeds.to = _to
    run_imitation_pass(world.trainer, batch, 1.0)
    assert len(calls) == 1
    assert calls[0][0] == ()
    assert calls[0][1]["device"] == batch.latents.device


def test_raise_restores_base_residency_and_dropout():
    shared = nn.Parameter(torch.tensor([0.25]))
    world = _build_world(shared, fail_turbo_on=True)
    with pytest.raises(RuntimeError, match="turbo_on failed"):
        run_imitation_pass(world.trainer, _batch(1), 1.0)
    assert world.sd._train_on_turbo is False
    assert world.residency.base == "cuda"
    assert world.residency.sampling == "cpu"
    assert world.events[-3:] == ["park_turbo", "flush", "host_base"]
    _assert_dropout_restored(world)
    world.sd.apply_turbo_teacher_mode.assert_not_called()
    assert world.trainer.network is world.base_net
    assert world.sd.network is world.base_net


def test_hook_one_step_sums_both_gradients(monkeypatch):
    import extensions_built_in.diffusion_models.z_image_diffsynth.trainer as trainer_mod

    printed = []

    def _print(*args, **kwargs):
        assert not events_started["events"]
        printed.append(args[0])

    monkeypatch.setattr(trainer_mod, "print_acc", _print)
    monkeypatch.setattr(
        trainer_mod.run_imitation_pass.__globals__["torch"],
        "randint",
        lambda low, high, size, **kwargs: torch.zeros(size, dtype=torch.long),
    )
    shared = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
    world = _build_world(shared, weight=2.0)
    events_started = {"events": world.events}
    opt = Adafactor([shared], lr=1e-4, weight_decay=0.0)
    timeline = []
    seen = {}
    real_zero = opt.zero_grad
    real_step = opt.step
    real_hook = opt.step_hook
    hook_calls = {"n": 0}

    def _zero(*args, **kwargs):
        timeline.append("zero")
        return real_zero(*args, **kwargs)

    def _step(closure=None):
        timeline.append("step")
        seen["grad"] = shared.grad.detach().clone()
        return real_step(closure)

    def _step_hook():
        hook_calls["n"] += 1
        return real_hook()

    opt.zero_grad = _zero
    opt.step = _step
    opt.step_hook = _step_hook

    def _concept(batch, microbatch_scale=1.0):
        timeline.append("concept")
        (shared * 3).sum().backward()
        return torch.tensor(5.0)

    real_backward = world.trainer.accelerator.backward

    def _backward(loss):
        timeline.append("turbo_on_backward")
        return real_backward(loss)

    world.trainer.accelerator.backward = _backward

    trainer = _bind_loop(world, opt, _concept)
    real_before = trainer.hook_before_optimizer_step

    def _before():
        timeline.append("hook")
        seen["network"] = trainer.network
        seen["flag"] = trainer.sd._train_on_turbo
        return real_before()

    trainer.hook_before_optimizer_step = _before
    batch = _batch(1)
    loss_dict = trainer.hook_train_loop([batch])
    assert timeline == ["zero", "concept", "turbo_on_backward", "hook", "step", "zero"]
    assert world.events[-8:] == [
        "park_base",
        "flush",
        "host_turbo",
        "turbo_off",
        "turbo_on_backward",
        "park_turbo",
        "flush",
        "host_base",
    ]
    assert loss_dict["loss"] == pytest.approx(5.0)
    assert loss_dict["turbo_lora"] == pytest.approx(0.0625)
    assert seen["grad"].shape == shared.shape
    assert float(seen["grad"]) == pytest.approx(4.0)
    assert opt.state[shared]["step"] == 1
    assert hook_calls["n"] == 1
    assert seen["network"] is world.base_net
    assert seen["flag"] is False
    assert printed == ["[zimage_diffsynth] turbo_lora=true weight=2 slots=8"]
    trainer.hook_train_loop([batch])
    assert printed == ["[zimage_diffsynth] turbo_lora=true weight=2 slots=8"]
    assert opt.state[shared]["step"] == 2


def test_hook_two_microbatches_weight_the_logged_mse(monkeypatch):
    import extensions_built_in.diffusion_models.z_image_diffsynth.turbo_lora.turbo_lora as turbo_lora

    slots = iter((0, 3))
    noises = {"n": 0}

    def _randint(low, high, size, **kwargs):
        return torch.tensor([next(slots)], dtype=torch.long)

    def _randn_like(tensor):
        noises["n"] += 1
        return torch.full_like(tensor, float(noises["n"]))

    monkeypatch.setattr(turbo_lora.torch, "randint", _randint)
    monkeypatch.setattr(turbo_lora.torch, "randn_like", _randn_like)
    shared = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
    world = _build_world(shared, weight=2.0)
    calls = []
    orig = type(world.sd).get_noise_prediction

    def _wrapped(self, latent_model_input, timestep, text_embeddings, batch=None, **kwargs):
        calls.append((latent_model_input, timestep.detach().clone()))
        return orig(
            self,
            latent_model_input,
            timestep,
            text_embeddings,
            batch=batch,
            **kwargs,
        )

    world.sd.get_noise_prediction = _wrapped.__get__(world.sd, type(world.sd))
    opt = Adafactor([shared], lr=1e-4, weight_decay=0.0)
    steps = {"n": 0}
    real_step = opt.step

    def _step(closure=None):
        steps["n"] += 1
        return real_step(closure)

    opt.step = _step
    losses = iter((10.0, 20.0))

    def _concept(batch, microbatch_scale=1.0):
        return torch.tensor(next(losses))

    trainer = _bind_loop(world, opt, _concept)
    batches = [_batch(1), _batch(3)]
    loss_dict = trainer.hook_train_loop(batches)
    assert loss_dict["loss"] == pytest.approx((10.0 * 1 + 20.0 * 3) / 4)
    logged = (0.0625 * 1 + 0.25 * 3) / 4
    assert loss_dict["turbo_lora"] == pytest.approx(logged)
    assert loss_dict["turbo_lora"] != pytest.approx((0.0625 + 0.25) / 2)
    assert loss_dict["turbo_lora"] != pytest.approx(logged * 2)
    assert steps["n"] == 1
    assert opt.state[shared]["step"] == 1
    assert noises["n"] == 2
    assert world.add_calls["n"] == 2
    first = calls[:4]
    second = calls[4:]
    assert len(first) == 4 and len(second) == 4
    assert len({id(item[0]) for item in first}) == 1
    assert len({id(item[0]) for item in second}) == 1
    assert id(first[0][0]) != id(second[0][0])
    assert torch.allclose(
        first[0][1],
        torch.full((1,), _CENTERS_8[0], dtype=first[0][1].dtype),
        atol=0.01,
    )
    assert torch.allclose(
        second[0][1],
        torch.full((3,), _CENTERS_8[3], dtype=second[0][1].dtype),
        atol=0.01,
    )
    assert not torch.allclose(first[0][1][:1], second[0][1][:1])


def test_accumulation_step_runs_pass_and_skips_optimizer_step():
    shared = nn.Parameter(torch.tensor([0.25]))
    world = _build_world(shared, weight=1.0)
    opt = SimpleNamespace(
        zero_grad=lambda *a, **k: None,
        step=MagicMock(),
        step_hook=MagicMock(),
    )

    def _concept(batch, microbatch_scale=1.0):
        return torch.tensor(1.0)

    trainer = _bind_loop(world, opt, _concept, accumulate=True)
    loss_dict = trainer.hook_train_loop([_batch(1)])
    opt.step.assert_not_called()
    opt.step_hook.assert_not_called()
    assert "turbo_on_backward" in world.events
    assert loss_dict["turbo_lora"] == pytest.approx(0.0625)
    assert loss_dict["loss"] == pytest.approx(1.0)


def test_bf16_lora_accumulates_until_one_step():
    if not torch.cuda.is_available():
        pytest.skip("stochastic bf16 accumulation rejects CPU tensors")
    shared = nn.Parameter(torch.tensor([0.25], device="cuda", dtype=torch.bfloat16))
    world = _build_world(shared, weight=1.0)
    opt = Adafactor([shared], lr=1e-4, weight_decay=0.0)
    hook_calls = {"n": 0}
    real_hook = opt.step_hook

    def _step_hook():
        hook_calls["n"] += 1
        return real_hook()

    opt.step_hook = _step_hook

    def _concept(batch, microbatch_scale=1.0):
        (shared * 3).sum().backward()
        assert shared.grad is None
        assert hasattr(shared, "_accum_grad")
        return torch.tensor(1.0)

    def _backward(loss):
        assert hook_calls["n"] == 0
        assert shared.grad is None
        assert hasattr(shared, "_accum_grad")
        loss.backward()
        world.events.append("turbo_on_backward")

    world.trainer.accelerator.backward = _backward
    trainer = _bind_loop(world, opt, _concept)
    trainer.hook_train_loop([_batch(1, device="cuda")])
    assert hook_calls["n"] == 1
    assert opt.state[shared]["step"] == 1
    assert not hasattr(shared, "_accum_grad")


def test_flag_false_leaves_parent_loop(monkeypatch):
    import extensions_built_in.diffusion_models.z_image_diffsynth.trainer as trainer_mod
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )

    def _boom(*args, **kwargs):
        raise AssertionError("imitation pass ran")

    monkeypatch.setattr(trainer_mod, "run_imitation_pass", _boom)
    printed = []
    monkeypatch.setattr(trainer_mod, "print_acc", lambda *a, **k: printed.append(a[0]))
    trainer = ZImageDiffSynthTrainer.__new__(ZImageDiffSynthTrainer)
    trainer.train_config = SimpleNamespace(
        turbo_lora=False,
        turbo_lora_weight="bad",
        optimizer="adafactor",
        turbo_prior_steps=8,
    )
    trainer.sd = SimpleNamespace(is_multistage=False)
    trainer.optimizer = SimpleNamespace(zero_grad=lambda *a, **k: None, step=MagicMock())
    trainer.train_single_accumulation = lambda batch, microbatch_scale=1.0: torch.tensor(4.0)
    trainer.steps_this_boundary = 0
    trainer.is_grad_accumulation_step = False
    trainer.model_config = SimpleNamespace(low_vram=False)
    trainer.embedding = None
    trainer.adapter = None
    trainer.ema = None
    trainer.lr_scheduler = SimpleNamespace(step=lambda: None)
    trainer.timer = Timer()
    trainer.network_config = SimpleNamespace(network_kwargs={"stable_lora_shrink": 0})
    trainer.network = _Net()
    loss_dict = trainer.hook_train_loop([_batch(2)])
    assert list(loss_dict.keys()) == ["loss"]
    assert loss_dict["loss"] == pytest.approx(4.0)
    assert batch_num_samples(_batch(2)) == 2
    trainer.optimizer.step.assert_called_once()
    assert printed == []
