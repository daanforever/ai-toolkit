"""Topology hooks for text-cache residency (plan step 4).

Covers default single-transformer path plus nonstandard owners:
z_image_diffsynth exclusive normal/turbo, z_image sampling DiT, Wan22 DualWan
(no-op ``.to``), Wan21 I2V ``image_encoder`` / low-VRAM preset early-return.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional, Tuple

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from toolkit.unloader import (
    collect_persistent_non_te_owners,
    enter_text_cache_residency,
    exit_text_cache_residency,
    unload_text_encoder,
)
from toolkit.util.device import devices_equal, safe_module_to_device

_HAS_CUDA = torch.cuda.is_available()
_CUDA = torch.device("cuda") if _HAS_CUDA else None
_CPU = torch.device("cpu")


def _param_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _all_on(module: nn.Module, device: torch.device) -> bool:
    for p in module.parameters():
        if not devices_equal(p.device, device):
            return False
    for b in module.buffers():
        if b is not None and not devices_equal(b.device, device):
            return False
    return True


def _make_te(device: torch.device) -> nn.Module:
    return nn.Linear(4, 4, bias=False).to(device)


# ---------------------------------------------------------------------------
# Default single-transformer (no model-specific override)
# ---------------------------------------------------------------------------


def _default_model(device: torch.device) -> SimpleNamespace:
    te = _make_te(device)
    unet = nn.Linear(4, 4, bias=False).to(device)
    vae = nn.Linear(4, 4, bias=False).to(device)
    network = nn.Linear(4, 4, bias=False).to(device)

    def text_encoder_to(dev, *a, **k):
        te.to(dev)

    return SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=device,
        torch_dtype=torch.float32,
        unet=unet,
        model=unet,
        vae=vae,
        network=network,
        _sampling_transformer=None,
        _sampling_network=None,
        adapter=None,
        refiner_unet=None,
        image_encoder=None,
        text_encoder_to=text_encoder_to,
        _train_on_turbo=False,
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_default_single_transformer_enter_te_only():
    assert _CUDA is not None
    model = _default_model(_CUDA)
    enter_text_cache_residency(model)
    assert _all_on(model.unet, _CPU)
    assert _all_on(model.network, _CPU)
    assert _all_on(model.vae, _CPU)
    assert _all_on(model.text_encoder, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_default_single_transformer_restore_main_on_train_device():
    assert _CUDA is not None
    model = _default_model(_CUDA)
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    exit_text_cache_residency(model)
    assert _all_on(model.unet, _CUDA)
    assert _all_on(model.network, _CUDA)


def test_extra_owner_hook_dedupes_aliases_by_object_id():
    shared = nn.Linear(2, 2)

    def iter_extra() -> Iterator[Tuple[str, nn.Module]]:
        yield "alias_shared", shared
        yield "other", nn.Linear(2, 2)

    model = SimpleNamespace(
        unet=shared,
        model=shared,
        vae=None,
        network=None,
        _sampling_transformer=None,
        _sampling_network=None,
        adapter=None,
        refiner_unet=None,
        image_encoder=None,
        iter_text_cache_extra_non_te_owners=iter_extra,
    )
    owners = collect_persistent_non_te_owners(model)
    # unet wins; model + alias_shared share id → dropped; other kept
    assert "unet" in owners
    assert "model" not in owners
    assert "alias_shared" not in owners
    assert "other" in owners
    assert len(owners) == 2


# ---------------------------------------------------------------------------
# z_image_diffsynth exclusive normal / turbo
# ---------------------------------------------------------------------------


class _TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)
        self.lora = nn.Parameter(torch.randn(4, 4), requires_grad=True)


class _TinyWrapper(nn.Module):
    def __init__(self, dit: nn.Module):
        super().__init__()
        self._inner_dit = dit

    def to(self, *args, **kwargs):
        self._inner_dit.to(*args, **kwargs)
        return self


class _ForceNet(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.adapter = nn.Parameter(torch.randn(2, 2), requires_grad=True)

    def force_to(self, device, dtype=None):
        self.to(device)
        if dtype is not None:
            self.adapter.data = self.adapter.data.to(dtype=dtype)


def _z_diffsynth_model(device: torch.device, *, train_on_turbo: bool) -> SimpleNamespace:
    raw = _TinyDiT().to(device)
    wrapper = _TinyWrapper(raw)
    sampling_inner = _TinyDiT().to(device)
    sampling = _TinyWrapper(sampling_inner).to(device)
    network = _ForceNet(raw).to(device)
    sampling_network = _ForceNet(sampling_inner).to(device)
    # Share trainable adapter identity like production share_parameters
    sampling_network.adapter = network.adapter
    te = _make_te(device)

    def place_training_dit(dev):
        target = torch.device(dev)
        raw.to(target)
        wrapper.to(target)

    def move_main_network(dev):
        target = torch.device(dev)
        if target.type == "cpu":
            return
        place_training_dit(target)
        network.force_to(target, torch.float32)

    def move_sampling_transformer(dev):
        sampling.to(dev)

    def apply_turbo_teacher_mode(enabled: bool):
        model._train_on_turbo = bool(enabled)
        if enabled:
            place_training_dit("cpu")
            move_sampling_transformer(device)
            sampling_network.force_to(device, torch.float32)
            model.network = sampling_network
        else:
            move_sampling_transformer("cpu")
            model.network = network
            move_main_network(device)

    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=device,
        torch_dtype=torch.float32,
        unet=wrapper,
        model=wrapper,
        _raw_dit=raw,
        vae=nn.Linear(2, 2).to(device),
        network=network,
        _sampling_transformer=sampling,
        _sampling_network=sampling_network,
        adapter=None,
        refiner_unet=None,
        image_encoder=None,
        text_encoder_to=lambda dev, *a, **k: te.to(dev),
        _place_training_dit=place_training_dit,
        _move_main_network=move_main_network,
        _move_sampling_transformer=move_sampling_transformer,
        apply_turbo_teacher_mode=apply_turbo_teacher_mode,
        _train_on_turbo=train_on_turbo,
    )
    return model


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_z_image_diffsynth_enter_parks_main_and_sampling():
    assert _CUDA is not None
    model = _z_diffsynth_model(_CUDA, train_on_turbo=False)
    enter_text_cache_residency(model)
    assert _all_on(model._raw_dit, _CPU)
    assert _all_on(model.model, _CPU)
    assert _all_on(model._sampling_transformer, _CPU)
    assert _all_on(model.network, _CPU)
    assert _all_on(model._sampling_network, _CPU)
    assert _all_on(model.text_encoder, _CUDA)
    assert not (
        _any_cuda(model._raw_dit) and _any_cuda(model._sampling_transformer)
    )


def _any_cuda(module: nn.Module) -> bool:
    return any(p.device.type == "cuda" for p in module.parameters())


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_z_image_diffsynth_normal_exit_main_active_sampling_cpu():
    assert _CUDA is not None
    model = _z_diffsynth_model(_CUDA, train_on_turbo=False)
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    exit_text_cache_residency(model)
    assert _all_on(model._raw_dit, _CUDA)
    assert _all_on(model._sampling_transformer, _CPU)
    assert not (_any_cuda(model._raw_dit) and _any_cuda(model._sampling_transformer))


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_z_image_diffsynth_turbo_exit_via_train_on_turbo():
    assert _CUDA is not None
    model = _z_diffsynth_model(_CUDA, train_on_turbo=True)
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    exit_text_cache_residency(model)
    assert _all_on(model._sampling_transformer, _CUDA)
    assert _all_on(model._raw_dit, _CPU)
    assert model.network is model._sampling_network
    assert not (_any_cuda(model._raw_dit) and _any_cuda(model._sampling_transformer))


# ---------------------------------------------------------------------------
# z_image sampling transformer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_z_image_sampling_transformer_enter_offload():
    assert _CUDA is not None
    te = _make_te(_CUDA)
    unet = nn.Linear(4, 4, bias=False).to(_CUDA)
    sampling = nn.Linear(4, 4, bias=False).to(_CUDA)
    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=_CUDA,
        torch_dtype=torch.float32,
        unet=unet,
        model=unet,
        vae=None,
        network=None,
        _sampling_transformer=sampling,
        _sampling_network=None,
        adapter=None,
        refiner_unet=None,
        image_encoder=None,
        text_encoder_to=lambda dev, *a, **k: te.to(dev),
        _train_on_turbo=False,
    )
    enter_text_cache_residency(model)
    assert _all_on(sampling, _CPU)
    assert _all_on(unet, _CPU)
    assert _all_on(te, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_z_image_sampling_transformer_normal_exit_stays_cpu():
    assert _CUDA is not None
    te = _make_te(_CUDA)
    unet = nn.Linear(4, 4, bias=False).to(_CUDA)
    sampling = nn.Linear(4, 4, bias=False).to(_CUDA)
    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=_CUDA,
        torch_dtype=torch.float32,
        unet=unet,
        model=unet,
        vae=None,
        network=None,
        _sampling_transformer=sampling,
        _sampling_network=None,
        adapter=None,
        refiner_unet=None,
        image_encoder=None,
        text_encoder_to=lambda dev, *a, **k: te.to(dev),
        _train_on_turbo=False,
    )
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    exit_text_cache_residency(model)
    assert _all_on(unet, _CUDA)
    assert _all_on(sampling, _CPU)


# ---------------------------------------------------------------------------
# Wan22 DualWan no-op .to()
# ---------------------------------------------------------------------------


class _FaithfulDualWan(nn.Module):
    """Tiny stand-in for DualWanTransformer3DModel (no-op ``.to``)."""

    def __init__(self):
        super().__init__()
        self.transformer_1 = nn.Linear(4, 4, bias=False)
        self.transformer_2 = nn.Linear(4, 4, bias=False)

    def to(self, *args, **kwargs):
        return self

    def _move_stages_to_device(self, device):
        target = torch.device(device)
        safe_module_to_device(self.transformer_1, target)
        safe_module_to_device(self.transformer_2, target)


def _wan22_model(device: torch.device) -> SimpleNamespace:
    te = _make_te(device)
    dual = _FaithfulDualWan().to(device)
    # Force stages on CUDA via safe path (dual.to is no-op)
    safe_module_to_device(dual.transformer_1, device)
    safe_module_to_device(dual.transformer_2, device)

    def place_training_dit(dev):
        dual._move_stages_to_device(dev)

    def move_main_network(dev):
        place_training_dit(dev)

    def iter_extra():
        yield "model.transformer_1", dual.transformer_1
        yield "model.transformer_2", dual.transformer_2

    return SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=device,
        torch_dtype=torch.float32,
        unet=dual,
        model=dual,
        vae=None,
        network=None,
        _sampling_transformer=None,
        _sampling_network=None,
        adapter=None,
        refiner_unet=None,
        image_encoder=None,
        text_encoder_to=lambda dev, *a, **k: te.to(dev),
        _place_training_dit=place_training_dit,
        _move_main_network=move_main_network,
        iter_text_cache_extra_non_te_owners=iter_extra,
        _train_on_turbo=False,
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_wan22_dual_enter_offloads_both_stages_despite_noop_to():
    assert _CUDA is not None
    model = _wan22_model(_CUDA)
    dual = model.model
    # Prove wrapper.to is a no-op (would leave CUDA if relied upon)
    dual.to("cpu")
    assert _param_device(dual.transformer_1).type == "cuda"
    assert _param_device(dual.transformer_2).type == "cuda"

    enter_text_cache_residency(model)
    assert _all_on(dual.transformer_1, _CPU), "transformer_1 must leave CUDA"
    assert _all_on(dual.transformer_2, _CPU), "transformer_2 must leave CUDA"
    assert _all_on(model.text_encoder, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_wan22_dual_restore_both_stages_on_train_device():
    assert _CUDA is not None
    model = _wan22_model(_CUDA)
    dual = model.model
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    exit_text_cache_residency(model)
    assert _all_on(dual.transformer_1, _CUDA)
    assert _all_on(dual.transformer_2, _CUDA)


# ---------------------------------------------------------------------------
# Wan21 I2V low-VRAM + image_encoder
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_wan21_i2v_enter_offloads_image_encoder():
    assert _CUDA is not None
    te = _make_te(_CUDA)
    unet = nn.Linear(4, 4, bias=False).to(_CUDA)
    ie = nn.Linear(4, 4, bias=False).to(_CUDA)
    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=_CUDA,
        torch_dtype=torch.float32,
        unet=unet,
        model=unet,
        vae=None,
        network=None,
        _sampling_transformer=None,
        _sampling_network=None,
        adapter=None,
        refiner_unet=None,
        image_encoder=ie,
        text_encoder_to=lambda dev, *a, **k: te.to(dev),
        _train_on_turbo=False,
        model_config=SimpleNamespace(low_vram=True),
    )
    enter_text_cache_residency(model)
    assert _all_on(ie, _CPU)
    assert _all_on(unet, _CPU)
    assert _all_on(te, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_wan21_i2v_low_vram_pre_enter_cpu_ie_stays_cpu_after_exit():
    """Production low_vram leaves IE on CPU before enter; snapshot keeps it CPU."""
    assert _CUDA is not None
    from toolkit.models.wan21.wan21_i2v import Wan21I2V

    te = _make_te(_CUDA)
    unet = nn.Linear(4, 4, bias=False).to(_CUDA)
    ie = nn.Linear(4, 4, bias=False).to(_CPU)

    sd = Wan21I2V.__new__(Wan21I2V)
    sd.text_encoder = te
    sd.pipeline = None
    sd.device_torch = _CUDA
    sd.torch_dtype = torch.float32
    sd.unet = unet
    sd.model = unet
    sd.vae = None
    sd.network = None
    sd._sampling_transformer = None
    sd._sampling_network = None
    sd.adapter = None
    sd.refiner_unet = None
    sd.image_encoder = ie
    sd._train_on_turbo = False
    sd.model_config = SimpleNamespace(low_vram=True)
    sd.text_encoder_to = lambda dev, *a, **k: te.to(dev)

    assert _param_device(ie).type == "cpu"
    enter_text_cache_residency(sd)
    assert _all_on(ie, _CPU)
    assert _all_on(unet, _CPU)
    assert _all_on(te, _CUDA)

    unload_text_encoder(sd)
    exit_text_cache_residency(sd)
    assert _all_on(unet, _CUDA)
    assert _all_on(ie, _CPU)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_wan21_i2v_non_low_vram_restore_remounts_image_encoder():
    assert _CUDA is not None
    te = _make_te(_CUDA)
    unet = nn.Linear(4, 4, bias=False).to(_CUDA)
    ie = nn.Linear(4, 4, bias=False).to(_CUDA)
    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=_CUDA,
        torch_dtype=torch.float32,
        unet=unet,
        model=unet,
        vae=None,
        network=None,
        _sampling_transformer=None,
        _sampling_network=None,
        adapter=None,
        refiner_unet=None,
        image_encoder=ie,
        text_encoder_to=lambda dev, *a, **k: te.to(dev),
        _train_on_turbo=False,
        model_config=SimpleNamespace(low_vram=False),
    )
    enter_text_cache_residency(model)
    assert _all_on(ie, _CPU)
    unload_text_encoder(model)
    exit_text_cache_residency(model)
    assert _all_on(ie, _CUDA)
    assert _all_on(unet, _CUDA)


# ---------------------------------------------------------------------------
# Real ZImageDiffSynthModel methods (not SimpleNamespace copies)
# ---------------------------------------------------------------------------


def _attach_fake_quant_payload(param: nn.Parameter, device: torch.device) -> None:
    param.qdata = torch.randn_like(param.data, device=device)
    param.scale = torch.tensor(1.0, device=device)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_z_image_diffsynth_real_set_device_state_skips_turbo_while_residency_active():
    """Real ``set_device_state`` must not remount turbo while text-cache residency is active."""
    assert _CUDA is not None
    from unittest.mock import MagicMock

    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model._train_on_turbo = True
    model._text_cache_residency_active = True
    model.vae = nn.Linear(2, 2)
    model.model = nn.Linear(2, 2)
    model.text_encoder = nn.Linear(2, 2)
    model.adapter = None
    model.refiner_unet = None
    apply = MagicMock()
    model.apply_turbo_teacher_mode = apply

    state = {
        "vae": {"training": False, "device": "cpu"},
        "unet": {"training": False, "requires_grad": False, "device": str(_CUDA)},
        "text_encoder": {"training": False, "requires_grad": False, "device": "cpu"},
        "adapter": {"training": False, "requires_grad": False, "device": "cpu"},
        "refiner_unet": {"training": False, "requires_grad": False, "device": "cpu"},
    }
    ZImageDiffSynthModel.set_device_state(model, state)
    apply.assert_not_called()
    # Forced CPU for base unet under turbo intent
    assert _param_device(model.unet).type == "cpu"

    model._text_cache_residency_active = False
    ZImageDiffSynthModel.set_device_state(model, state)
    apply.assert_called_once_with(True)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_z_image_diffsynth_real_place_training_dit_enter_moves_quant_payload():
    """Enter via real ``_place_training_dit`` + ``_DiTUnetWrapper`` relocates fake quant payload."""
    assert _CUDA is not None
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
        _DiTUnetWrapper,
    )
    from toolkit.util.device import quantized_payload_device

    class _TinyFrozenDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)

    dit = _TinyFrozenDiT().to(_CUDA)
    _attach_fake_quant_payload(dit.weight, _CUDA)
    assert devices_equal(quantized_payload_device(dit.weight), _CUDA)

    wrapper = _DiTUnetWrapper(dit)
    te = _make_te(_CUDA)

    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = _CUDA
    model.torch_dtype = torch.float32
    model._raw_dit = dit
    model.model = wrapper
    model.vae = nn.Linear(2, 2).to(_CUDA)
    model.network = None
    model._sampling_transformer = None
    model._sampling_network = None
    model.adapter = None
    model.refiner_unet = None
    model.image_encoder = None
    model.text_encoder = te
    model.pipeline = None
    model._train_on_turbo = False
    model.text_encoder_to = lambda *a, **k: te.to(*a, **k)

    enter_text_cache_residency(model)

    assert devices_equal(dit.weight.device, _CPU)
    payload = quantized_payload_device(dit.weight)
    assert payload is not None
    assert devices_equal(payload, _CPU)
    assert devices_equal(dit.weight.scale.device, _CPU)
    assert _all_on(te, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_z_image_diffsynth_real_place_training_dit_compiled_blocks_offload(
    monkeypatch,
):
    """Real ``_place_training_dit`` must use ``move_dit_with_compiled_blocks`` and unwrap."""
    assert _CUDA is not None
    from extensions_built_in.diffusion_models.z_image_diffsynth import compile_blocks
    from extensions_built_in.diffusion_models.z_image_diffsynth.compile_blocks import (
        move_dit_with_compiled_blocks as _real_move,
    )
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
        _DiTUnetWrapper,
    )
    from toolkit.util.device import quantized_payload_device

    class _FakeCompiled(nn.Module):
        def __init__(self, orig: nn.Module):
            super().__init__()
            self._orig_mod = orig

        def forward(self, *args, **kwargs):
            return self._orig_mod(*args, **kwargs)

    class _TinyLayeredDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
            for p in self.parameters():
                p.requires_grad_(False)

    dit = _TinyLayeredDiT().to(_CUDA)
    orig0 = dit.layers[0]
    orig1 = dit.layers[1]
    _attach_fake_quant_payload(next(orig0.parameters()), _CUDA)
    dit.layers[0] = _FakeCompiled(orig0)
    dit.layers[1] = _FakeCompiled(orig1)
    assert hasattr(dit.layers[0], "_orig_mod")

    helper_calls: list = []

    def _spy_move(dit_mod, block_names=None, *args, **kwargs):
        helper_calls.append(True)
        return _real_move(dit_mod, block_names, *args, **kwargs)

    monkeypatch.setattr(compile_blocks, "move_dit_with_compiled_blocks", _spy_move)

    wrapper = _DiTUnetWrapper(dit)
    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = _CUDA
    model.torch_dtype = torch.float32
    model._raw_dit = dit
    model.model = wrapper

    # Call the real method while still on CUDA so the helper does a real move (not early no-op).
    need = ZImageDiffSynthModel._place_training_dit(model, "cpu")

    assert need is True
    assert helper_calls, "_place_training_dit must invoke move_dit_with_compiled_blocks"
    assert not hasattr(dit.layers[0], "_orig_mod"), "compiled blocks must unwrap on offload"
    assert dit.layers[0] is orig0
    assert dit.layers[1] is orig1
    assert _all_on(dit, _CPU)
    payload = quantized_payload_device(next(orig0.parameters()))
    assert payload is not None
    assert devices_equal(payload, _CPU)


# ---------------------------------------------------------------------------
# Audited secondary owners (assistant_adapter / taesd / ARA / decorator / ...)
# ---------------------------------------------------------------------------


class _TinyForceToNetwork(nn.Module):
    """Minimal LoRASpecial-like network with force_to."""

    def __init__(self):
        super().__init__()
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.unet_loras = [self.lora_A]

    def force_to(self, device, dtype):
        self.to(device)
        for lora in self.unet_loras:
            lora.to(device, dtype)


def _secondary_owners_model(device: torch.device) -> SimpleNamespace:
    te = _make_te(device)
    unet = nn.Linear(4, 4, bias=False).to(device)
    assistant_adapter = nn.Linear(4, 4, bias=False).to(device)
    taesd = nn.Linear(4, 4, bias=False).to(device)
    decorator = nn.Linear(4, 4, bias=False).to(device)
    audio_processor = nn.Linear(4, 4, bias=False).to(device)
    assistant_lora = _TinyForceToNetwork().to(device)
    accuracy_recovery_adapter = _TinyForceToNetwork().to(device)

    def text_encoder_to(dev, *a, **k):
        te.to(dev)

    return SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=device,
        torch_dtype=torch.float32,
        unet=unet,
        model=unet,
        vae=None,
        network=None,
        _sampling_transformer=None,
        _sampling_network=None,
        adapter=None,
        refiner_unet=None,
        image_encoder=None,
        assistant_adapter=assistant_adapter,
        taesd=taesd,
        decorator=decorator,
        audio_processor=audio_processor,
        assistant_lora=assistant_lora,
        accuracy_recovery_adapter=accuracy_recovery_adapter,
        text_encoder_to=text_encoder_to,
        _train_on_turbo=False,
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_secondary_owners_enter_cpu_exit_remount_policy_initial():
    """Initial: remount ARA/decorator/assistant_adapter/taesd; keep assistant_lora/audio CPU."""
    assert _CUDA is not None
    model = _secondary_owners_model(_CUDA)
    remount = (
        model.assistant_adapter,
        model.taesd,
        model.decorator,
        model.accuracy_recovery_adapter,
    )
    stay_cpu = (model.audio_processor, model.assistant_lora)
    enter_text_cache_residency(model)
    for mod in remount + stay_cpu:
        assert _all_on(mod, _CPU)
    assert _all_on(model.text_encoder, _CUDA)
    assert _all_on(model.unet, _CPU)

    unload_text_encoder(model)
    exit_text_cache_residency(model)
    for mod in remount:
        assert _all_on(mod, _CUDA)
    for mod in stay_cpu:
        assert _all_on(mod, _CPU)
    assert _all_on(model.unet, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for topology residency")
def test_secondary_owners_runtime_recache_remount_policy():
    """Runtime: park all audited owners; remount only train-needed secondaries."""
    assert _CUDA is not None
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

    model = _secondary_owners_model(_CUDA)
    unload_text_encoder(model)
    # Mid-train: backbone + remount-policy owners on CUDA; TE Fake/stashed.
    model.unet.to(_CUDA)
    for name in ("assistant_adapter", "taesd", "decorator", "audio_processor"):
        getattr(model, name).to(_CUDA)
    model.assistant_lora.force_to(_CUDA, torch.float32)
    model.accuracy_recovery_adapter.force_to(_CUDA, torch.float32)

    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.device_torch = _CUDA
    t.is_caching_text_embeddings = True
    t.train_config = SimpleNamespace(unload_text_encoder=False)
    t.sd = model
    seen: list = []

    def cache_spy():
        seen.append("cache")
        assert _all_on(model.assistant_adapter, _CPU)
        assert _all_on(model.taesd, _CPU)
        assert _all_on(model.decorator, _CPU)
        assert _all_on(model.audio_processor, _CPU)
        assert _all_on(model.assistant_lora, _CPU)
        assert _all_on(model.accuracy_recovery_adapter, _CPU)
        assert bool(getattr(model, "_text_cache_residency_active", False))

    t.cache_sample_prompts = cache_spy
    t._recache_sample_prompts_runtime = (
        DiffusionTrainer._recache_sample_prompts_runtime.__get__(t, DiffusionTrainer)
    )
    t._recache_sample_prompts_runtime()

    assert seen == ["cache"]
    assert _all_on(model.assistant_adapter, _CUDA)
    assert _all_on(model.taesd, _CUDA)
    assert _all_on(model.decorator, _CUDA)
    assert _all_on(model.accuracy_recovery_adapter, _CUDA)
    assert _all_on(model.audio_processor, _CPU)
    assert _all_on(model.assistant_lora, _CPU)
    assert not bool(getattr(model, "_text_cache_residency_active", False))


def test_collect_includes_audited_secondary_owners_by_behavior():
    """Observable collect: audited secondary Modules are returned (id-deduped)."""
    model = SimpleNamespace(
        unet=None,
        model=None,
        vae=None,
        network=None,
        _sampling_transformer=None,
        _sampling_network=None,
        adapter=None,
        refiner_unet=None,
        image_encoder=None,
        assistant_adapter=nn.Linear(2, 2),
        taesd=nn.Linear(2, 2),
        assistant_lora=nn.Linear(2, 2),
        accuracy_recovery_adapter=nn.Linear(2, 2),
        decorator=nn.Linear(2, 2),
        audio_processor=nn.Linear(2, 2),
    )
    owners = collect_persistent_non_te_owners(model)
    for name in (
        "assistant_adapter",
        "taesd",
        "assistant_lora",
        "accuracy_recovery_adapter",
        "decorator",
        "audio_processor",
    ):
        assert name in owners
        assert isinstance(owners[name], nn.Module)
