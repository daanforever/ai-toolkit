"""Regression: classic LoRA/LoKr torch_multiplier must follow weight device.

After text-cache residency, LoRA is created on CPU then remounted to CUDA.
``torch_multiplier`` is a plain tensor (not a Parameter), so ``force_to`` and
shared-parameter skip paths can leave it on CPU while LoRA output is on CUDA.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Tuple

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from toolkit.lora_special import LoRASpecialNetwork
from toolkit.unloader import (
    _TEXT_CACHE_NETWORK_OWNER_ATTRS,
    _move_network_owner_to_device,
    _refresh_network_torch_multipliers,
)
from toolkit.util.device import devices_equal

_HAS_CUDA = torch.cuda.is_available()
_CUDA = torch.device("cuda") if _HAS_CUDA else None
_CPU = torch.device("cpu")


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


def _build_classic_lora(
    *,
    network_type: str = "lora",
    ephemeral: bool = False,
) -> Tuple[LoRASpecialNetwork, nn.Module]:
    from toolkit.config_modules import NetworkConfig

    text_enc = nn.Module()
    unet = UNet2DConditionModel()
    network_config = None
    if network_type.lower() == "lokr":
        network_config = NetworkConfig(type="lokr", linear=2, linear_alpha=1.0, lokr_factor=-1)
    net = LoRASpecialNetwork(
        text_encoder=text_enc,
        unet=unet,
        train_text_encoder=False,
        train_unet=True,
        lora_dim=2,
        alpha=1.0,
        multiplier=1.0,
        network_type=network_type,
        ephemeral_lora=ephemeral,
        network_config=network_config,
        target_lin_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE,
        target_conv_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3,
    )
    assert net.unet_loras
    net.apply_to(None, unet, False, True)
    return net, unet


def _first_lora_weight_device(network: LoRASpecialNetwork) -> torch.device:
    mod = network.get_all_modules()[0]
    if hasattr(mod, "lora_down"):
        return mod.lora_down.weight.device
    if hasattr(mod, "lokr_w1"):
        return mod.lokr_w1.device
    if hasattr(mod, "lokr_w1_a"):
        return mod.lokr_w1_a.device
    raise AssertionError("unknown LoRA module type")


def test_force_to_refreshes_torch_multiplier_device():
    """force_to must move torch_multiplier with weights (no manual refresh)."""
    net, _ = _build_classic_lora()
    net._update_torch_multiplier()
    assert devices_equal(net.torch_multiplier.device, _CPU)

    target = _CUDA if _HAS_CUDA else _CPU
    if not _HAS_CUDA:
        # Still exercise the refresh path on CPU (idempotent).
        net.force_to(_CPU, torch.float32)
        assert devices_equal(net.torch_multiplier.device, _first_lora_weight_device(net))
        return

    net.force_to(target, torch.float32)
    assert devices_equal(_first_lora_weight_device(net), target)
    assert devices_equal(
        net.torch_multiplier.device, target
    ), "force_to left torch_multiplier on stale device"


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required to expose stale CPU multiplier")
def test_shared_sampling_skips_force_to_but_multiplier_follows_weights():
    """Production rule: remount main only; sampling shares params — multiplier must still match."""
    main, _ = _build_classic_lora()
    sampling, _ = _build_classic_lora(ephemeral=True)
    sampling.share_parameters_with(main)
    main._update_torch_multiplier()
    sampling._update_torch_multiplier()
    assert devices_equal(main.torch_multiplier.device, _CPU)
    assert devices_equal(sampling.torch_multiplier.device, _CPU)

    # Simulate unloader: move main to CUDA; sampling force_to skipped (shared already CUDA).
    _move_network_owner_to_device("network", main, _CUDA, model=None)
    # Early-return path: sampling params already on CUDA via share.
    _move_network_owner_to_device("_sampling_network", sampling, _CUDA, model=None)

    model = SimpleNamespace(network=main, _sampling_network=sampling)
    _refresh_network_torch_multipliers(model)

    assert devices_equal(_first_lora_weight_device(main), _CUDA)
    assert devices_equal(_first_lora_weight_device(sampling), _CUDA)
    assert main.unet_loras[0].lora_down.weight is sampling.unet_loras[0].lora_down.weight
    assert devices_equal(main.torch_multiplier.device, _CUDA)
    assert devices_equal(
        sampling.torch_multiplier.device, _CUDA
    ), "sampling torch_multiplier stayed on CPU after shared remount"


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for device-mismatch forward")
def test_lora_forward_with_cpu_multiplier_on_cuda_weights():
    """Forward must succeed even if torch_multiplier is still on CPU (PEFT-style cast)."""
    net, unet = _build_classic_lora()
    unet.to(_CUDA)
    net.force_to(_CUDA, torch.float32)
    # Intentionally leave / put multiplier on CPU (pre-fix / setter no-op scenario).
    net.torch_multiplier = torch.tensor((1.0,), device=_CPU, dtype=torch.float32)
    assert net._multiplier == 1.0
    net.multiplier = 1.0  # setter no-op — must not be the only refresh path
    assert devices_equal(net.torch_multiplier.device, _CPU)

    net.is_active = True
    x = torch.randn(1, 4, device=_CUDA, dtype=torch.float32)
    out = unet.block.linear(x)
    assert out.device.type == "cuda"
    assert devices_equal(out.device, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for device-mismatch forward")
def test_lokr_forward_with_cpu_multiplier_on_cuda_weights():
    """LoKr path must cast multiplier to weight device before multiply."""
    net, unet = _build_classic_lora(network_type="lokr")
    unet.to(_CUDA)
    net.force_to(_CUDA, torch.float32)
    net.torch_multiplier = torch.tensor((1.0,), device=_CPU, dtype=torch.float32)
    net.is_active = True
    x = torch.randn(1, 4, device=_CUDA, dtype=torch.float32)
    out = unet.block.linear(x)
    assert devices_equal(out.device, _CUDA)


def test_refresh_network_torch_multipliers_covers_attrs():
    """Helper must visit every network owner attr that exposes _update_torch_multiplier."""
    main, _ = _build_classic_lora()
    sampling, _ = _build_classic_lora(ephemeral=True)
    sampling.share_parameters_with(main)
    main._update_torch_multiplier()
    sampling._update_torch_multiplier()

    # Stale multipliers on a fake "wrong" device tensor while weights stay on CPU.
    main.torch_multiplier = torch.tensor((1.0,), device=_CPU)
    sampling.torch_multiplier = torch.tensor((2.0,), device=_CPU)

    model = SimpleNamespace(
        network=main,
        _sampling_network=sampling,
        assistant_lora=None,
        accuracy_recovery_adapter=None,
    )
    assert "network" in _TEXT_CACHE_NETWORK_OWNER_ATTRS
    assert "_sampling_network" in _TEXT_CACHE_NETWORK_OWNER_ATTRS
    _refresh_network_torch_multipliers(model)

    assert devices_equal(main.torch_multiplier.device, _first_lora_weight_device(main))
    assert devices_equal(
        sampling.torch_multiplier.device, _first_lora_weight_device(sampling)
    )
    # Refresh rebuilds from _multiplier, not from the stale tensor value.
    assert float(main.torch_multiplier.reshape(-1)[0].item()) == pytest.approx(1.0)
