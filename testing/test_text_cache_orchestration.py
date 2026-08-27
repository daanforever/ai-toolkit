"""Orchestration contracts for text-cache residency (step-scoped).

Step 3 (cpu-network-init): when ``is_caching_text_embeddings`` is True, main
and sampling LoRA/PEFT are created/applied on CPU without premature CUDA
remount of wrapped base or adapter params. Optimizer receives the same
Parameter objects; train-device remount happens later via common lifecycle exit.

Step 5 (initial migration): dataset disk cache + pre-train sample/unconditional
encodes share one long enter phase; VAE/adapter/unet prepare remount only after
unload+exit; encode failures abort without partial train restore.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.peft_network import PeftNetwork
from toolkit.prompt_utils import PromptEmbeds
from toolkit.unloader import (
    FakeTextEncoder,
    abort_text_cache_residency,
    enter_text_cache_residency,
    exit_text_cache_residency,
    unload_text_encoder,
)
from toolkit.util.device import devices_equal

_HAS_CUDA = torch.cuda.is_available()
_CUDA = torch.device("cuda") if _HAS_CUDA else None
_CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Tiny DiT / LoRA fixtures (CPU-only, no checkpoints)
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    def __init__(self, d: int = 4):
        super().__init__()
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)


class _FeedForward(nn.Module):
    def __init__(self, d: int = 4):
        super().__init__()
        self.w1 = nn.Linear(d, d * 2, bias=False)
        self.w2 = nn.Linear(d * 2, d, bias=False)


class _ZImageTransformerBlock(nn.Module):
    def __init__(self, d: int = 4):
        super().__init__()
        self.attention = _Attention(d)
        self.feed_forward = _FeedForward(d)


class _InnerDiT(nn.Module):
    def __init__(self, d: int = 4, n_blocks: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([_ZImageTransformerBlock(d) for _ in range(n_blocks)])

    def forward(self, x):
        for blk in self.layers:
            x = blk.attention.to_q(x) + blk.feed_forward.w2(
                torch.nn.functional.silu(blk.feed_forward.w1(x))
            )
        return x


class _UnetWrapper(nn.Module):
    def __init__(self, dit: nn.Module):
        super().__init__()
        self._inner_dit = dit

    def forward(self, *args, **kwargs):
        return self._inner_dit(*args, **kwargs)


class _LoRACompatibleLinear(nn.Linear):
    pass


class UNet2DConditionModel(nn.Module):
    """Parent class name matches LoRASpecialNetwork default target modules."""

    def __init__(self, d: int = 4):
        super().__init__()
        self.frozen = nn.Linear(d, d, bias=False)
        self.frozen.weight.requires_grad_(False)
        self.block = nn.Module()
        self.block.linear = _LoRACompatibleLinear(d, d, bias=False)


class _StubBase:
    arch = "zimage_diffsynth"
    target_lora_modules = ["_ZImageTransformerBlock"]

    def convert_lora_weights_before_save(self, sd):
        return sd

    def convert_lora_weights_before_load(self, sd):
        return sd


def _bare_process(
    *,
    caching: bool,
    train_device: torch.device,
    dtype: str = "fp32",
) -> BaseSDTrainProcess:
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
    proc.is_caching_text_embeddings = caching
    proc.device_torch = train_device
    proc.network_config = None
    proc.train_config = SimpleNamespace(dtype=dtype)
    return proc


def _adapter_params(network: Any) -> List[nn.Parameter]:
    if hasattr(network, "prepare_optimizer_params"):
        groups = network.prepare_optimizer_params(
            text_encoder_lr=1e-4, unet_lr=1e-4, default_lr=1e-4
        )
        out: List[nn.Parameter] = []
        for g in groups:
            out.extend(g["params"] if isinstance(g, dict) else g)
        return out
    return [p for p in network.parameters() if p.requires_grad]


def _all_params_on(module: nn.Module, device: torch.device) -> bool:
    return all(devices_equal(p.device, device) for p in module.parameters())


def _build_classic_lora_pair() -> Tuple[LoRASpecialNetwork, LoRASpecialNetwork, nn.Module, nn.Module]:
    text_enc = nn.Module()
    main_unet = UNet2DConditionModel()
    sampling_unet = UNet2DConditionModel()
    common = dict(
        text_encoder=text_enc,
        train_text_encoder=False,
        train_unet=True,
        lora_dim=2,
        alpha=1.0,
        target_lin_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE,
        target_conv_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3,
    )
    main = LoRASpecialNetwork(unet=main_unet, **common)
    sampling = LoRASpecialNetwork(unet=sampling_unet, ephemeral_lora=True, **common)
    assert main.unet_loras and sampling.unet_loras
    return main, sampling, main_unet, sampling_unet


def _build_peft_pair(network_type: str = "peft") -> Tuple[PeftNetwork, PeftNetwork, nn.Module, nn.Module]:
    base = _StubBase()
    main_unet = _UnetWrapper(_InnerDiT())
    sampling_unet = _UnetWrapper(_InnerDiT())
    common = dict(
        text_encoder=None,
        multiplier=1.0,
        lora_dim=2,
        alpha=2.0,
        train_unet=True,
        train_text_encoder=False,
        network_type=network_type,
        base_model=base,
        target_lin_modules=base.target_lora_modules,
    )
    main = PeftNetwork(unet=main_unet, **common)
    sampling = PeftNetwork(unet=sampling_unet, ephemeral_lora=True, **common)
    assert main.unet_loras and sampling.unet_loras
    return main, sampling, main_unet, sampling_unet


def _place_like_basesd(
    main: Any,
    sampling: Any,
    *,
    caching: bool,
    train_device: torch.device,
) -> Tuple[torch.device, List[nn.Parameter], torch.optim.Optimizer]:
    """Main force_to via production seam → share → apply → optimizer (no sampling force_to)."""
    proc = _bare_process(caching=caching, train_device=train_device)
    proc.network = main
    proc._force_initial_network_to_device()
    init_device = proc._resolve_initial_network_device()
    sampling.share_parameters_with(main)
    if hasattr(sampling, "_update_torch_multiplier"):
        sampling._update_torch_multiplier()
    if hasattr(main, "apply_to"):
        main.apply_to(None, None, False, True)
    if hasattr(sampling, "apply_to"):
        sampling.apply_to(None, None, False, True)
    params = _adapter_params(main)
    opt = torch.optim.SGD(params, lr=1e-3)
    return init_device, params, opt


# ---------------------------------------------------------------------------
# Device selection / production force_to seam
# ---------------------------------------------------------------------------


def test_resolve_initial_network_device_caching_is_cpu():
    train = torch.device("cuda:0") if _HAS_CUDA else torch.device("cpu")
    proc = _bare_process(caching=True, train_device=train)
    assert devices_equal(proc._resolve_initial_network_device(), _CPU)


def test_resolve_initial_network_device_non_caching_is_train_device():
    train = torch.device("cuda:0") if _HAS_CUDA else torch.device("cpu")
    proc = _bare_process(caching=False, train_device=train)
    assert devices_equal(proc._resolve_initial_network_device(), train)


@pytest.mark.parametrize("caching,expected", [(True, "cpu"), (False, "train")])
def test_force_initial_network_to_device_passes_resolved_device(caching, expected):
    """Runtime seam: ``_force_initial_network_to_device`` calls network.force_to with resolved device."""
    train = torch.device("cuda:0") if _HAS_CUDA else torch.device("cpu")
    proc = _bare_process(caching=caching, train_device=train, dtype="fp32")
    force_calls: List[Tuple[torch.device, torch.dtype]] = []

    class _SpyNetwork:
        def force_to(self, device, dtype):
            force_calls.append((torch.device(device), dtype))

    proc.network = _SpyNetwork()
    proc._force_initial_network_to_device()

    assert len(force_calls) == 1
    got_device, got_dtype = force_calls[0]
    want = _CPU if expected == "cpu" else train
    assert devices_equal(got_device, want)
    assert got_dtype == torch.float


# ---------------------------------------------------------------------------
# Classic LoRA + PEFT/DoRA placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder,network_type",
    [
        (_build_classic_lora_pair, "lora"),
        (_build_peft_pair, "peft"),
        (lambda: _build_peft_pair("peft_dora"), "peft_dora"),
    ],
    ids=["classic_lora", "peft", "peft_dora"],
)
def test_caching_keeps_main_sampling_base_and_adapters_on_cpu(builder, network_type):
    main, sampling, main_base, sampling_base = builder()
    init_device, params, opt = _place_like_basesd(
        main, sampling, caching=True, train_device=_CUDA or _CPU
    )

    assert devices_equal(init_device, _CPU)
    assert _all_params_on(main_base, _CPU)
    assert _all_params_on(sampling_base, _CPU)
    for p in params:
        assert devices_equal(p.device, _CPU), f"{network_type} adapter param not on CPU"
    for p in _adapter_params(sampling):
        assert devices_equal(p.device, _CPU)

    # PEFT must not remount wrapped base to CUDA when caching (force_to got CPU).
    if isinstance(main, PeftNetwork):
        assert _all_params_on(main.peft_model, _CPU)
        assert _all_params_on(sampling.peft_model, _CPU)


@pytest.mark.parametrize(
    "builder",
    [_build_classic_lora_pair, _build_peft_pair, lambda: _build_peft_pair("peft_dora")],
    ids=["classic_lora", "peft", "peft_dora"],
)
def test_caching_share_parameters_preserves_is_identity(builder):
    main, sampling, _, _ = builder()
    _place_like_basesd(main, sampling, caching=True, train_device=_CUDA or _CPU)

    main_params = _adapter_params(main)
    sampling_params = _adapter_params(sampling)
    assert main_params and sampling_params
    # Shared trainable weights must be the same objects (sampling adopts main).
    main_by_id = {id(p): p for p in main_params}
    shared = [p for p in sampling_params if id(p) in main_by_id]
    assert shared, "expected shared Parameter identity after share_parameters_with"
    for p in shared:
        assert p is main_by_id[id(p)]


@pytest.mark.parametrize(
    "builder",
    [_build_classic_lora_pair, _build_peft_pair],
    ids=["classic_lora", "peft"],
)
def test_caching_optimizer_holds_original_parameter_objects(builder):
    main, sampling, _, _ = builder()
    _, params, opt = _place_like_basesd(
        main, sampling, caching=True, train_device=_CUDA or _CPU
    )
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    param_ids = {id(p) for p in params}
    assert opt_ids == param_ids
    # Same objects the network still exposes.
    assert {id(p) for p in _adapter_params(main)} == param_ids


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required to distinguish train device")
@pytest.mark.parametrize(
    "builder",
    [_build_classic_lora_pair, _build_peft_pair],
    ids=["classic_lora", "peft"],
)
def test_non_caching_force_to_uses_train_device(builder):
    main, sampling, main_base, _ = builder()
    init_device, params, _ = _place_like_basesd(
        main, sampling, caching=False, train_device=_CUDA
    )
    assert devices_equal(init_device, _CUDA)
    for p in params:
        assert devices_equal(p.device, _CUDA)
    # Main base moves with PEFT force_to; classic LoRA modules move, frozen base
    # may stay where created — adapter params are the contract for both.
    if isinstance(main, PeftNetwork):
        assert _all_params_on(main.peft_model, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
@pytest.mark.parametrize(
    "builder",
    [_build_classic_lora_pair, _build_peft_pair],
    ids=["classic_lora", "peft"],
)
def test_non_caching_leaves_shared_params_on_train_device_without_sampling_force_to(builder):
    """Sampling must not be force_to(CPU) after share — that would yank shared train-device params."""
    main, sampling, _, _ = builder()
    force_calls: List[Tuple[str, Any]] = []
    orig = sampling.force_to

    def spy(device, dtype):
        force_calls.append((str(torch.device(device)), dtype))
        return orig(device, dtype)

    setattr(sampling, "force_to", spy)
    _place_like_basesd(main, sampling, caching=False, train_device=_CUDA)
    shared = _adapter_params(sampling)[0]
    assert devices_equal(shared.device, _CUDA)
    assert force_calls == [], (
        "placement must not call sampling.force_to; "
        f"got {force_calls!r} (would move shared params off train device)"
    )


# ---------------------------------------------------------------------------
# Optimizer identity survives common lifecycle restore (reuse Step-2 seam)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA lifecycle restore needs CUDA")
@pytest.mark.parametrize("use_peft", [False, True], ids=["lora", "peft"])
def test_optimizer_identity_survives_lifecycle_after_cpu_init(use_peft):
    """Caching CPU init → optimizer → enter/unload/exit keeps Parameter identity."""
    if use_peft:
        main, sampling, main_base, sampling_base = _build_peft_pair()
    else:
        main, sampling, main_base, sampling_base = _build_classic_lora_pair()

    _, params, opt = _place_like_basesd(
        main, sampling, caching=True, train_device=_CUDA
    )
    param_ids_before = {id(p) for p in params}
    opt_ids_before = {id(p) for g in opt.param_groups for p in g["params"]}
    shared_main = params[0]
    shared_sampling = _adapter_params(sampling)[0]
    assert shared_sampling is shared_main

    te = nn.Linear(4, 4, bias=False).to(_CUDA)
    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=_CUDA,
        torch_dtype=torch.float32,
        unet=main_base,
        vae=nn.Linear(4, 4, bias=False),
        network=main,
        _sampling_transformer=sampling_base,
        _sampling_network=sampling,
        adapter=None,
        refiner_unet=None,
        text_encoder_to=lambda dev, *a, **k: te.to(dev),
        _move_main_network=lambda dev: (
            main_base.to(dev),
            main.force_to(dev, torch.float32),
        ),
        _move_sampling_transformer=lambda dev: sampling_base.to(dev),
        _train_on_turbo=False,
    )

    enter_text_cache_residency(model)
    unload_text_encoder(model)
    exit_text_cache_residency(model)

    assert shared_sampling is shared_main
    assert {id(p) for p in _adapter_params(main)} == param_ids_before
    assert {id(p) for g in opt.param_groups for p in g["params"]} == opt_ids_before


# ---------------------------------------------------------------------------
# Step 5 — initial disk + pre-train sample cache (one long residency phase)
# ---------------------------------------------------------------------------


def _make_embeds() -> PromptEmbeds:
    return PromptEmbeds(torch.zeros(1, 4))


class _TrackModule(nn.Module):
    def __init__(self, d: int = 4):
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)
        self._to_calls: List[str] = []

    def to(self, *args, **kwargs):
        device = None
        if args and isinstance(args[0], (torch.device, str)):
            device = torch.device(args[0])
        elif "device" in kwargs:
            device = torch.device(kwargs["device"])
        if device is not None:
            self._to_calls.append(str(device))
        return super().to(*args, **kwargs)


def _any_param_on(module: nn.Module, device: torch.device) -> bool:
    return any(devices_equal(p.device, device) for p in module.parameters())


def _build_residency_sd(*, train_device: torch.device) -> SimpleNamespace:
    te = _TrackModule().to(train_device)
    unet = _TrackModule().to(_CPU)
    vae = _TrackModule().to(_CPU)
    adapter = _TrackModule().to(_CPU)
    network = _TrackModule().to(_CPU)

    def text_encoder_to(dev, *a, **k):
        live = model.text_encoder
        if isinstance(live, nn.Module) and not isinstance(live, FakeTextEncoder):
            live.to(dev)
        real = getattr(model, "_real_text_encoder", None)
        if isinstance(real, nn.Module):
            real.to(dev)

    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=train_device,
        torch_dtype=torch.float32,
        device=str(train_device),
        unet=unet,
        vae=vae,
        network=network,
        adapter=adapter,
        refiner_unet=None,
        _sampling_transformer=None,
        _sampling_network=None,
        noise_scheduler=SimpleNamespace(),
        encode_control_in_text_embeddings=False,
        has_multiple_control_images=False,
        text_encoder_to=text_encoder_to,
        _place_training_dit=lambda dev: unet.to(dev) or True,
        _move_main_network=lambda dev: (unet.to(dev), network.to(dev)),
        _move_sampling_transformer=lambda _dev: None,
        _train_on_turbo=False,
        encode_prompt=lambda *a, **k: _make_embeds(),
        set_device_state_preset=MagicMock(),
    )
    return model


def _build_sdtrainer_stub(
    *,
    caching: bool,
    latents_cached: bool,
    train_device: torch.device,
    unload_only: bool = False,
) -> Any:
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    sd = _build_residency_sd(train_device=train_device)
    order: List[str] = []
    prepare_calls: List[Any] = []

    class _Accel:
        even_batches = True

        def prepare(self, obj, device_placement=None):
            if device_placement is not None:
                raise AssertionError(
                    "deferred prepare must use normal Accelerate path "
                    f"(device_placement={device_placement!r})"
                )
            prepare_calls.append(obj)
            order.append(f"prepare:{type(obj).__name__}")
            return obj

    t = SDTrainer.__new__(SDTrainer)
    t.sd = sd
    t.device_torch = train_device
    t.is_caching_text_embeddings = caching
    t.is_latents_cached = latents_cached
    t.modules_being_trained = []
    t.accelerator = _Accel()
    t.adapter = sd.adapter
    t.datasets = None
    t.datasets_reg = None
    t.data_loader = None
    t.trigger_word = "tok"
    t.do_long_prompts = False
    t.cached_blank_embeds = None
    t.unconditional_embeds = None
    t.cached_trigger_embeds = None
    t._accelerator_prepare_deferred = False
    t._accelerator_prepare_done = False
    t.train_config = SimpleNamespace(
        unconditional_prompt="",
        do_prior_divergence=False,
        unload_text_encoder=unload_only and not caching,
        train_text_encoder=False,
        train_refiner=False,
        diff_output_preservation=False,
        diff_output_preservation_class="",
        negative_prompt=None,
        blank_prompt_preservation=False,
        diffusion_feature_extractor_path=None,
        gradient_checkpointing=False,
        disable_sampling=True,
    )
    t.sample_config = None

    # Production BaseSDTrainProcess.hook_before_train_loop → prepare_accelerator.
    def _super_hook():
        order.append("prepare_accelerator")
        BaseSDTrainProcess.prepare_accelerator(t)

    # Bind real SDTrainer / Base helpers; stub only the Base super call.
    t.hook_before_train_loop = SDTrainer.hook_before_train_loop.__get__(t, SDTrainer)
    t._remount_vae_after_text_cache = SDTrainer._remount_vae_after_text_cache.__get__(
        t, SDTrainer
    )
    t._remount_adapter_after_text_cache = SDTrainer._remount_adapter_after_text_cache.__get__(
        t, SDTrainer
    )
    t.prepare_accelerator = BaseSDTrainProcess.prepare_accelerator.__get__(
        t, BaseSDTrainProcess
    )
    t.finalize_accelerator_prepare = BaseSDTrainProcess.finalize_accelerator_prepare.__get__(
        t, BaseSDTrainProcess
    )
    t._apply_accelerator_prepare = BaseSDTrainProcess._apply_accelerator_prepare.__get__(
        t, BaseSDTrainProcess
    )
    t.cache_sample_prompts = lambda: order.append("cache_sample_prompts")
    t.optimizer = torch.optim.SGD([p for p in sd.network.parameters()], lr=1e-3)
    t.lr_scheduler = SimpleNamespace(name="lr_sched")
    t.adapter_config = SimpleNamespace(train=True)

    # Intercept enter/exit/unload via wrappers that record order, then call real.
    _enter = enter_text_cache_residency
    _exit = exit_text_cache_residency
    _unload = unload_text_encoder

    import extensions_built_in.sd_trainer.SDTrainer as sdt_mod

    def enter_spy(model, device=None):
        was_active = bool(getattr(model, "_text_cache_residency_active", False))
        order.append("enter")
        te_before = list(sd.text_encoder._to_calls)
        unet_before = list(sd.unet._to_calls)
        _enter(model, device)
        if was_active:
            order.append("enter_noop")
            assert sd.text_encoder._to_calls == te_before
            assert sd.unet._to_calls == unet_before
        else:
            order.append("enter_moved")

    def exit_spy(model, device=None):
        order.append("exit")
        _exit(model, device)

    def unload_spy(model):
        order.append("unload")
        _unload(model)

    t._order = order
    t._prepare_calls = prepare_calls
    t._super_hook = _super_hook
    t._enter_spy = enter_spy
    t._exit_spy = exit_spy
    t._unload_spy = unload_spy
    t._sdt_mod = sdt_mod
    return t


def _run_hook_with_spies(t: Any, monkeypatch: pytest.MonkeyPatch) -> List[str]:
    monkeypatch.setattr(t._sdt_mod, "enter_text_cache_residency", t._enter_spy)
    monkeypatch.setattr(t._sdt_mod, "exit_text_cache_residency", t._exit_spy)
    monkeypatch.setattr(t._sdt_mod, "unload_text_encoder", t._unload_spy)
    monkeypatch.setattr(
        BaseSDTrainProcess, "hook_before_train_loop", lambda self: t._super_hook()
    )

    def snr_spy(*a, **k):
        t._order.append("snr_setup")

    monkeypatch.setattr(t._sdt_mod, "add_all_snr_to_noise_scheduler", snr_spy)
    # encode_prompt recording
    orig_encode = t.sd.encode_prompt

    def encode_spy(*a, **k):
        t._order.append("encode")
        # VAE must stay CPU throughout TE-only when caching
        if t.is_caching_text_embeddings and not isinstance(
            t.sd.text_encoder, FakeTextEncoder
        ):
            assert not _any_param_on(t.sd.vae, t.device_torch) or devices_equal(
                t.device_torch, _CPU
            ), "VAE remounted during TE-only encode"
            assert not _any_param_on(t.sd.unet, t.device_torch) or devices_equal(
                t.device_torch, _CPU
            ), "unet remounted during TE-only encode"
            assert "snr_setup" not in t._order, "SNR must not run during TE-only encode"
        return orig_encode(*a, **k)

    t.sd.encode_prompt = encode_spy
    t.hook_before_train_loop()
    return t._order


def test_cache_text_embeddings_enters_before_encode(tmp_path, monkeypatch):
    """Initial disk cache: enter before first encode; residency stays active."""
    from toolkit.dataloader_mixins import TextEmbeddingCachingMixin
    import toolkit.dataloader_mixins as dlm

    train = _CUDA or _CPU
    sd = _build_residency_sd(train_device=train)
    calls: List[str] = []

    def enter_spy(model, device=None):
        calls.append("enter")
        enter_text_cache_residency(model, device)

    monkeypatch.setattr(dlm, "enter_text_cache_residency", enter_spy)

    emb_path = tmp_path / "item_te.safetensors"
    file_item = SimpleNamespace(
        path=str(tmp_path / "item.png"),
        caption="a cat",
        encode_control_in_text_embeddings=False,
        control_path=None,
        dataset_config=SimpleNamespace(
            shuffle_tokens_keep=1,
            shuffle_tokens_split_re=None,
            shuffle_tokens_join=None,
        ),
        latent_load_device=None,
        is_text_embedding_cached=False,
        get_text_embedding_path=lambda recalculate=False: str(emb_path),
    )

    def encode_spy(caption, **kwargs):
        calls.append("encode")
        assert bool(getattr(sd, "_text_cache_residency_active", False))
        assert "enter" in calls
        return _make_embeds()

    sd.encode_prompt = encode_spy
    sd.set_device_state_preset = MagicMock(
        side_effect=AssertionError("device-state preset must not drive TE cache")
    )

    class _DS:
        def __init__(self):
            self.dataset_path = str(tmp_path)
            self.train_config = SimpleNamespace(steps=1)
            self.dataset_config = SimpleNamespace(
                shuffle_tokens=False,
                shuffle_tokens_cap=4,
                caption_dropout_rate=0,
                caption_dropout_keep=0,
            )
            self.file_list = [file_item]
            self.sd = sd

        def __len__(self):
            return 1

    TextEmbeddingCachingMixin.cache_text_embeddings(_DS())
    assert calls[0] == "enter"
    assert "encode" in calls
    assert bool(getattr(sd, "_text_cache_residency_active", False)), (
        "residency must stay active after dataset cache"
    )
    sd.set_device_state_preset.assert_not_called()


def test_multi_dataset_cache_one_phase_exit_once_in_pretrain(monkeypatch):
    """Two dataset enters share one phase; pre-train enter is no-op; one exit."""
    train = _CUDA or _CPU
    t = _build_sdtrainer_stub(
        caching=True, latents_cached=True, train_device=train
    )
    # Simulate prior dataset caches (first enter moves, second is no-op).
    enter_text_cache_residency(t.sd)
    assert bool(getattr(t.sd, "_text_cache_residency_active", False))
    te_calls_after_first = list(t.sd.text_encoder._to_calls)
    enter_text_cache_residency(t.sd)  # second dataset
    assert t.sd.text_encoder._to_calls == te_calls_after_first

    order = _run_hook_with_spies(t, monkeypatch)
    assert order.count("enter") == 1
    assert "enter_noop" in order
    assert order.count("exit") == 1
    assert order.count("unload") == 1
    assert order.index("unload") < order.index("exit")
    assert not bool(getattr(t.sd, "_text_cache_residency_active", False))


@pytest.mark.parametrize("latents_cached,vae_expect", [(True, "cpu"), (False, "train")])
def test_vae_cpu_during_te_only_remount_after_exit(latents_cached, vae_expect, monkeypatch):
    train = _CUDA or _CPU
    t = _build_sdtrainer_stub(
        caching=True, latents_cached=latents_cached, train_device=train
    )
    enter_text_cache_residency(t.sd)
    assert not _any_param_on(t.sd.vae, train) or devices_equal(train, _CPU)

    vae_devices_during_encode: List[str] = []
    orig_encode = t.sd.encode_prompt

    def encode_and_snap(*a, **k):
        vae_dev = next(t.sd.vae.parameters()).device
        vae_devices_during_encode.append(str(vae_dev))
        return orig_encode(*a, **k)

    t.sd.encode_prompt = encode_and_snap
    _run_hook_with_spies(t, monkeypatch)

    for d in vae_devices_during_encode:
        assert devices_equal(torch.device(d), _CPU) or devices_equal(train, _CPU)

    final_vae = next(t.sd.vae.parameters()).device
    if vae_expect == "cpu" or devices_equal(train, _CPU):
        assert devices_equal(final_vae, _CPU)
    else:
        assert devices_equal(final_vae, train)

    # Adapter remount after exit (present on stub).
    final_adapter = next(t.sd.adapter.parameters()).device
    if devices_equal(train, _CPU):
        assert devices_equal(final_adapter, _CPU)
    else:
        assert devices_equal(final_adapter, train)


def test_snr_setup_after_exit_when_caching(monkeypatch):
    train = _CUDA or _CPU
    t = _build_sdtrainer_stub(
        caching=True, latents_cached=True, train_device=train
    )
    enter_text_cache_residency(t.sd)
    order = _run_hook_with_spies(t, monkeypatch)
    assert "snr_setup" in order
    assert order.index("exit") < order.index("snr_setup"), (
        f"scheduler SNR must run after exit; order={order}"
    )


def test_deferred_prepare_zero_before_exit_once_after_finalize_noop(monkeypatch):
    train = _CUDA or _CPU
    t = _build_sdtrainer_stub(
        caching=True, latents_cached=True, train_device=train
    )
    # Trainable adapter so finalize prepares it.
    t.adapter_config = SimpleNamespace(train=True)
    enter_text_cache_residency(t.sd)

    # Early prepare_accelerator alone must not prepare modules/optimizer.
    BaseSDTrainProcess.prepare_accelerator(t)
    assert t._accelerator_prepare_deferred is True
    assert t._prepare_calls == []
    assert t._accelerator_prepare_done is False

    order = _run_hook_with_spies(t, monkeypatch)
    exit_i = order.index("exit")
    assert all(
        (not x.startswith("prepare:")) or (i > exit_i) for i, x in enumerate(order)
    ), f"no prepare before exit; order={order}"

    # Each necessary object prepared exactly once after exit.
    assert t._prepare_calls.count(t.sd.vae) == 1
    assert t._prepare_calls.count(t.sd.unet) == 1
    assert t._prepare_calls.count(t.sd.network) == 1
    assert t._prepare_calls.count(t.adapter) == 1
    assert t._prepare_calls.count(t.optimizer) == 1
    assert t._prepare_calls.count(t.lr_scheduler) == 1
    assert t._accelerator_prepare_done is True

    # Finalize again is a no-op.
    before = list(t._prepare_calls)
    t.finalize_accelerator_prepare()
    assert t._prepare_calls == before


def test_encode_error_aborts_without_partial_train_restore(monkeypatch):
    train = _CUDA or _CPU
    t = _build_sdtrainer_stub(
        caching=True, latents_cached=False, train_device=train
    )
    enter_text_cache_residency(t.sd)

    def boom(*a, **k):
        raise RuntimeError("encode/cache failed")

    t.sd.encode_prompt = boom
    monkeypatch.setattr(
        BaseSDTrainProcess, "hook_before_train_loop", lambda self: t._super_hook()
    )
    monkeypatch.setattr(
        t._sdt_mod,
        "add_all_snr_to_noise_scheduler",
        lambda *a, **k: None,
    )

    with pytest.raises(RuntimeError, match="encode/cache failed"):
        t.hook_before_train_loop()

    # No TE + backbone co-residency; no partial train remount of unet to train device
    te = t.sd.text_encoder
    if isinstance(te, FakeTextEncoder):
        te_cuda = False
    else:
        te_cuda = _HAS_CUDA and _any_param_on(te, _CUDA)
    unet_cuda = _HAS_CUDA and _any_param_on(t.sd.unet, _CUDA)
    assert not (te_cuda and unet_cuda)
    if _HAS_CUDA and not devices_equal(train, _CPU):
        assert not _any_param_on(t.sd.unet, _CUDA), "abort must not remount train backbone"
        assert not _any_param_on(t.sd.vae, _CUDA), "abort must not remount VAE"
    assert not bool(getattr(t.sd, "_text_cache_residency_active", False))
    assert t._prepare_calls == [], "encode failure must not finalize prepare"


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for stashed TE coresidency case")
def test_abort_moves_stashed_real_te_when_fake_attached():
    """Fake live + stashed real TE left on CUDA must be CPU after abort."""
    assert _CUDA is not None
    sd = _build_residency_sd(train_device=_CUDA)
    enter_text_cache_residency(sd)
    real_te = sd.text_encoder
    # Simulate unload that installed Fake but left stash on CUDA (failed/no-op move).
    sd._real_text_encoder = real_te
    sd.text_encoder = FakeTextEncoder(device=_CUDA, dtype=torch.float32)
    real_te.to(_CUDA)
    assert _any_param_on(real_te, _CUDA)
    setattr(sd, "_text_cache_residency_active", True)

    abort_text_cache_residency(sd)

    assert not bool(getattr(sd, "_text_cache_residency_active", False))
    assert devices_equal(next(real_te.parameters()).device, _CPU)
    assert devices_equal(next(sd.unet.parameters()).device, _CPU)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for prepare-failure cleanup")
def test_prepare_failure_after_exit_aborts_no_coresidency(monkeypatch):
    """Failure during finalize/remount after unload+exit: original error, no TE+backbone CUDA."""
    assert _CUDA is not None
    t = _build_sdtrainer_stub(
        caching=True, latents_cached=False, train_device=_CUDA
    )
    enter_text_cache_residency(t.sd)

    monkeypatch.setattr(t._sdt_mod, "enter_text_cache_residency", t._enter_spy)
    monkeypatch.setattr(t._sdt_mod, "exit_text_cache_residency", t._exit_spy)
    monkeypatch.setattr(t._sdt_mod, "unload_text_encoder", t._unload_spy)
    monkeypatch.setattr(
        BaseSDTrainProcess, "hook_before_train_loop", lambda self: t._super_hook()
    )
    monkeypatch.setattr(
        t._sdt_mod, "add_all_snr_to_noise_scheduler", lambda *a, **k: None
    )

    real_finalize = t.finalize_accelerator_prepare

    def boom_finalize():
        real_finalize()
        # Leave a stashed TE on CUDA to prove abort cleans stash after partial success.
        stash = getattr(t.sd, "_real_text_encoder", None)
        if isinstance(stash, nn.Module):
            stash.to(_CUDA)
        raise RuntimeError("prepare/remount failed")

    t.finalize_accelerator_prepare = boom_finalize

    with pytest.raises(RuntimeError, match="prepare/remount failed") as ei:
        t.hook_before_train_loop()

    # Original error surfaces (cleanup may be chained via __cause__).
    assert "prepare/remount failed" in str(ei.value)

    stash = getattr(t.sd, "_real_text_encoder", None)
    if isinstance(stash, nn.Module):
        assert devices_equal(next(stash.parameters()).device, _CPU)
    te_cuda = False
    live = t.sd.text_encoder
    if isinstance(live, nn.Module) and not isinstance(live, FakeTextEncoder):
        te_cuda = _any_param_on(live, _CUDA)
    if isinstance(stash, nn.Module):
        te_cuda = te_cuda or _any_param_on(stash, _CUDA)
    backbone_cuda = _any_param_on(t.sd.unet, _CUDA) or _any_param_on(t.sd.network, _CUDA)
    assert not (te_cuda and backbone_cuda)
    assert not bool(getattr(t.sd, "_text_cache_residency_active", False))


def test_unload_only_path_skips_enter_exit(monkeypatch):
    """Mode without cache_text_embeddings: unload-only keeps legacy TE→CPU, no residency."""
    train = _CUDA or _CPU
    t = _build_sdtrainer_stub(
        caching=False, latents_cached=False, train_device=train, unload_only=True
    )
    order = _run_hook_with_spies(t, monkeypatch)
    assert "enter" not in order
    assert "exit" not in order
    assert "unload" not in order  # Fake unload not used
    # Non-caching prepares immediately and sets SNR before unload-only TE offload.
    assert "snr_setup" in order
    assert t._prepare_calls, "non-caching must prepare during prepare_accelerator"
    # VAE remounted for non-caching path (when train != cpu)
    if not devices_equal(train, _CPU):
        assert devices_equal(next(t.sd.vae.parameters()).device, train)
    assert not isinstance(t.sd.text_encoder, FakeTextEncoder)
    assert devices_equal(next(t.sd.text_encoder.parameters()).device, _CPU)


def test_abort_text_cache_residency_offloads_without_restore():
    train = _CUDA or _CPU
    sd = _build_residency_sd(train_device=train)
    enter_text_cache_residency(sd)
    assert bool(getattr(sd, "_text_cache_residency_active", False))
    if _HAS_CUDA:
        assert _any_param_on(sd.text_encoder, train)
    abort_text_cache_residency(sd)
    assert not bool(getattr(sd, "_text_cache_residency_active", False))
    assert devices_equal(next(sd.text_encoder.parameters()).device, _CPU)
    assert devices_equal(next(sd.unet.parameters()).device, _CPU)


# ---------------------------------------------------------------------------
# Pre-dataset module placement (assistant_adapter / taesd / decorator)
# ---------------------------------------------------------------------------


def _all_params_on(module: nn.Module, device: torch.device) -> bool:
    return all(devices_equal(p.device, device) for p in module.parameters())


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for placement policy")
def test_decorator_caching_place_cpu_exit_remounts_cuda():
    """Caching: decorator created on resolver CPU; exit remounts to train CUDA."""
    assert _CUDA is not None
    proc = _bare_process(caching=True, train_device=_CUDA)
    place = proc._resolve_initial_network_device()
    assert devices_equal(place, _CPU)
    decorator = nn.Linear(4, 4, bias=False).to(place, dtype=torch.float32)
    sd = _build_residency_sd(train_device=_CUDA)
    sd.decorator = decorator
    assert _all_params_on(decorator, _CPU)

    enter_text_cache_residency(sd)
    assert _all_params_on(decorator, _CPU)
    unload_text_encoder(sd)
    exit_text_cache_residency(sd)
    assert _all_params_on(decorator, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for placement policy")
def test_assistant_adapter_taesd_caching_place_cpu_exit_remounts_cuda():
    """Caching: trainer-owned modules placed via resolver CPU; exit remounts CUDA."""
    assert _CUDA is not None
    proc = _bare_process(caching=True, train_device=_CUDA)
    place = proc._resolve_initial_network_device()
    assert devices_equal(place, _CPU)

    sd = _build_residency_sd(train_device=_CUDA)
    assistant = nn.Linear(4, 4, bias=False).to(place)
    taesd = nn.Linear(4, 4, bias=False).to(place)
    # Mirror as before_dataset_load does after CPU create.
    sd.assistant_adapter = assistant
    sd.taesd = taesd
    assert _all_params_on(assistant, _CPU)
    assert _all_params_on(taesd, _CPU)

    enter_text_cache_residency(sd)
    assert _all_params_on(assistant, _CPU)
    assert _all_params_on(taesd, _CPU)
    unload_text_encoder(sd)
    exit_text_cache_residency(sd)
    assert _all_params_on(assistant, _CUDA)
    assert _all_params_on(taesd, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for placement policy")
def test_before_dataset_load_uses_resolver_device_for_assistant_and_taesd(monkeypatch):
    """Production before_dataset_load places assistant_adapter/taesd on resolver device."""
    assert _CUDA is not None
    from extensions_built_in.sd_trainer import SDTrainer as sdt_mod
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    t = SDTrainer.__new__(SDTrainer)
    t.device_torch = _CUDA
    t.is_caching_text_embeddings = True
    t.sd = SimpleNamespace()
    t.assistant_adapter = None
    t.taesd = None
    t.train_config = SimpleNamespace(
        adapter_assist_name_or_path="dummy/t2i",
        adapter_assist_type="t2i",
        train_turbo=True,
        show_turbo_outputs=True,
        dtype="fp32",
    )
    t.model_config = SimpleNamespace(is_xl=False)
    t._resolve_initial_network_device = (
        BaseSDTrainProcess._resolve_initial_network_device.__get__(t, SDTrainer)
    )

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(2, 2))

        def eval(self):
            return self

        def requires_grad_(self, *_a, **_k):
            return self

    def _from_pretrained(*_a, **_k):
        return _Tiny()

    monkeypatch.setattr(sdt_mod.T2IAdapter, "from_pretrained", staticmethod(_from_pretrained))
    monkeypatch.setattr(
        sdt_mod.AutoencoderTiny, "from_pretrained", staticmethod(_from_pretrained)
    )

    SDTrainer.before_dataset_load(t)
    assert t.assistant_adapter is not None
    assert t.taesd is not None
    assert _all_params_on(t.assistant_adapter, _CPU)
    assert _all_params_on(t.taesd, _CPU)
    assert t.sd.assistant_adapter is t.assistant_adapter
    assert t.sd.taesd is t.taesd

    t.is_caching_text_embeddings = False
    t.assistant_adapter = None
    t.taesd = None
    SDTrainer.before_dataset_load(t)
    assert _all_params_on(t.assistant_adapter, _CUDA)
    assert _all_params_on(t.taesd, _CUDA)


# ---------------------------------------------------------------------------
# ConceptSliderTrainer: common enter before concept encodes
# ---------------------------------------------------------------------------


def _build_concept_slider_sd(*, train_device: torch.device) -> SimpleNamespace:
    """SD with unet/network/VAE/sampling/adapter initially on train_device."""
    te = nn.Linear(4, 4, bias=False).to(train_device)
    unet = nn.Linear(4, 4, bias=False).to(train_device)
    vae = nn.Linear(4, 4, bias=False).to(train_device)
    network = nn.Linear(4, 4, bias=False).to(train_device)
    adapter = nn.Linear(4, 4, bias=False).to(train_device)
    sampling = nn.Linear(4, 4, bias=False).to(train_device)

    def text_encoder_to(dev, *a, **k):
        live = model.text_encoder
        if isinstance(live, nn.Module) and not isinstance(live, FakeTextEncoder):
            live.to(dev)
        real = getattr(model, "_real_text_encoder", None)
        if isinstance(real, nn.Module):
            real.to(dev)

    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=train_device,
        torch_dtype=torch.float32,
        unet=unet,
        model=unet,
        vae=vae,
        network=network,
        adapter=adapter,
        refiner_unet=None,
        _sampling_transformer=sampling,
        _sampling_network=None,
        text_encoder_to=text_encoder_to,
        _place_training_dit=lambda dev: unet.to(dev) or True,
        _move_main_network=lambda dev: (unet.to(dev), network.to(dev)),
        _move_sampling_transformer=lambda dev: sampling.to(dev),
        _train_on_turbo=False,
    )
    return model


def _assert_te_only_residency(sd: SimpleNamespace, train_device: torch.device) -> None:
    assert bool(getattr(sd, "_text_cache_residency_active", False))
    assert _all_params_on(sd.text_encoder, train_device)
    for name in ("unet", "network", "vae", "adapter", "_sampling_transformer"):
        assert _all_params_on(getattr(sd, name), _CPU), f"{name} must be CPU during encode"


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for ConceptSlider residency")
def test_concept_slider_caching_enter_before_encode_parent_sees_active():
    """Caching: enter before concept encodes; non-TE owners CPU; parent sees active."""
    assert _CUDA is not None
    from extensions_built_in.concept_slider.ConceptSliderTrainer import (
        ConceptSliderTrainer,
    )

    sd = _build_concept_slider_sd(train_device=_CUDA)
    encode_n = {"n": 0}

    def encode_spy(prompts, **kwargs):
        encode_n["n"] += 1
        _assert_te_only_residency(sd, _CUDA)
        return _make_embeds()

    sd.encode_prompt = encode_spy

    t = ConceptSliderTrainer.__new__(ConceptSliderTrainer)
    t.sd = sd
    t.device_torch = _CUDA
    t.is_caching_text_embeddings = True
    t.positive_prompt = "pos"
    t.negative_prompt = "neg"
    t.target_class = "tgt"
    t.anchor_class = "anc"
    t.positive_prompt_embeds = None
    t.negative_prompt_embeds = None
    t.target_class_embeds = None
    t.anchor_class_embeds = None
    parent_saw_active = {"v": False}

    def parent_hook():
        parent_saw_active["v"] = bool(
            getattr(t.sd, "_text_cache_residency_active", False)
        )

    # Bind production ConceptSlider hook; stub only DiffusionTrainer parent.
    t.hook_before_train_loop = ConceptSliderTrainer.hook_before_train_loop.__get__(
        t, ConceptSliderTrainer
    )
    # Parent chain: ConceptSlider -> DiffusionTrainer -> SDTrainer.
    # Patch DiffusionTrainer.hook_before_train_loop used via super().
    import extensions_built_in.sd_trainer.DiffusionTrainer as dt_mod

    orig_parent = dt_mod.DiffusionTrainer.hook_before_train_loop

    def _parent(self):
        parent_hook()

    try:
        dt_mod.DiffusionTrainer.hook_before_train_loop = _parent
        t.hook_before_train_loop()
    finally:
        dt_mod.DiffusionTrainer.hook_before_train_loop = orig_parent

    assert encode_n["n"] == 4
    assert parent_saw_active["v"] is True
    assert t.positive_prompt_embeds is not None
    assert t.anchor_class_embeds is not None
    # Still active — parent stub did not unload/exit (production parent owns that).
    assert bool(getattr(sd, "_text_cache_residency_active", False))


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for ConceptSlider abort")
def test_concept_slider_encode_failure_aborts_before_parent():
    """Encode failure before parent: abort cleanup; original error propagates."""
    assert _CUDA is not None
    from extensions_built_in.concept_slider.ConceptSliderTrainer import (
        ConceptSliderTrainer,
    )
    import extensions_built_in.sd_trainer.DiffusionTrainer as dt_mod

    sd = _build_concept_slider_sd(train_device=_CUDA)
    calls = {"encode": 0, "parent": 0}

    def encode_boom(prompts, **kwargs):
        calls["encode"] += 1
        if calls["encode"] == 2:
            raise RuntimeError("concept encode boom")
        _assert_te_only_residency(sd, _CUDA)
        return _make_embeds()

    sd.encode_prompt = encode_boom

    t = ConceptSliderTrainer.__new__(ConceptSliderTrainer)
    t.sd = sd
    t.device_torch = _CUDA
    t.is_caching_text_embeddings = True
    t.positive_prompt = "pos"
    t.negative_prompt = "neg"
    t.target_class = "tgt"
    t.anchor_class = None
    t.positive_prompt_embeds = None
    t.negative_prompt_embeds = None
    t.target_class_embeds = None
    t.anchor_class_embeds = None
    t.hook_before_train_loop = ConceptSliderTrainer.hook_before_train_loop.__get__(
        t, ConceptSliderTrainer
    )

    orig_parent = dt_mod.DiffusionTrainer.hook_before_train_loop

    def _parent(self):
        calls["parent"] += 1

    try:
        dt_mod.DiffusionTrainer.hook_before_train_loop = _parent
        with pytest.raises(RuntimeError, match="concept encode boom") as ei:
            t.hook_before_train_loop()
    finally:
        dt_mod.DiffusionTrainer.hook_before_train_loop = orig_parent

    assert calls["parent"] == 0
    assert calls["encode"] == 2
    assert not bool(getattr(sd, "_text_cache_residency_active", False))
    assert _all_params_on(sd.unet, _CPU)
    assert _all_params_on(sd.network, _CPU)
    assert _all_params_on(sd.vae, _CPU)
    assert _all_params_on(sd.adapter, _CPU)
    assert _all_params_on(sd._sampling_transformer, _CPU)
    # Cleanup chained when abort itself fails is covered elsewhere; here abort succeeds.
    assert ei.value.__cause__ is None or "concept encode boom" in str(ei.value)
