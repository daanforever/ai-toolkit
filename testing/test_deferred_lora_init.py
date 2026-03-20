"""Tests for deferred LoRA init and finalize hook."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

import toolkit.lora_special as lora_special_mod
from toolkit.lora_special import LoRAModule
from toolkit.lora_utils.deferred_lora_init import (
    finalize_deferred_lora_init,
    lora_down_present_in_loaded_keys,
)


class _StubLoRANetwork:
    """Minimal stand-in for LoRASpecialNetwork (weakref-capable, unlike SimpleNamespace)."""

    def __init__(self, **kwargs):
        self.network_type = kwargs.pop("network_type", "lora")
        self.ephemeral_lora = kwargs.pop("ephemeral_lora", False)
        self.deferred_lora_init = kwargs.pop("deferred_lora_init", False)
        self.network_config = kwargs.pop("network_config", None)
        for k, v in kwargs.items():
            setattr(self, k, v)


def _linear_network(**kwargs):
    return _StubLoRANetwork(**kwargs)


def test_lora_down_present_exact_and_prefix_collision():
    name = "lora_unet_blocks_0"
    assert lora_down_present_in_loaded_keys(
        name,
        frozenset({f"{name}.lora_down.weight"}),
    )
    assert not lora_down_present_in_loaded_keys(
        name,
        frozenset({f"{name}_other.lora_down.weight"}),
    )


def test_finalize_noop_when_flag_false():
    net = SimpleNamespace(deferred_lora_init=False)
    finalize_deferred_lora_init(net, frozenset())


def test_finalize_invokes_module_hook():
    class _L:
        def __init__(self):
            self.calls = []

        def finalize_deferred_lora_init_if_needed(self, ks):
            self.calls.append(ks)

    te = _L()
    unet = _L()
    net = SimpleNamespace(
        deferred_lora_init=True,
        text_encoder_loras=[te],
        unet_loras=[unet],
    )
    keys = frozenset({"k"})
    finalize_deferred_lora_init(net, keys)
    assert te.calls == [keys]
    assert unet.calls == [keys]


def test_finalize_idempotent_second_pass_no_extra_pissa_calls():
    org = torch.nn.Linear(8, 16, bias=False)
    nk = SimpleNamespace(network_kwargs={"init_lora_weights": "pissa"})
    net = _linear_network(
        deferred_lora_init=True,
        network_config=nk,
    )
    mod = LoRAModule("layer.foo", org, network=net, lora_dim=4)
    net.text_encoder_loras = []
    net.unet_loras = [mod]
    assert mod._deferred_lora_init_pending is True
    down_before = mod.lora_down.weight.detach().clone()

    with patch.object(
        lora_special_mod,
        "try_init_linear_lora_down_pissa",
        wraps=lora_special_mod.try_init_linear_lora_down_pissa,
    ) as mocked:
        finalize_deferred_lora_init(net, frozenset())
        assert mocked.call_count == 1
        assert mod._deferred_lora_init_pending is False
        assert not torch.equal(down_before, mod.lora_down.weight)

        finalize_deferred_lora_init(net, frozenset())
        assert mocked.call_count == 1


def test_loramodule_ephemeral_zeros_weights():
    org = torch.nn.Linear(6, 10, bias=False)
    net = _linear_network(ephemeral_lora=True)
    mod = LoRAModule("layer.bar", org, network=net, lora_dim=3)
    assert (mod.lora_down.weight == 0).all()
    assert (mod.lora_up.weight == 0).all()
    assert mod._deferred_lora_init_pending is False


def test_loramodule_deferred_pissa_skips_pissa_in_init_sets_pending():
    org = torch.nn.Linear(6, 10, bias=False)
    nk = SimpleNamespace(network_kwargs={"init_lora_weights": "pissa"})
    net = _linear_network(deferred_lora_init=True, network_config=nk)
    with patch("toolkit.lora_special.try_init_linear_lora_down_pissa") as mocked:
        mod = LoRAModule("layer.baz", org, network=net, lora_dim=3)
        mocked.assert_not_called()
    assert mod._deferred_lora_init_pending is True


def test_finalize_clears_pending_when_checkpoint_has_lora_down():
    org = torch.nn.Linear(6, 10, bias=False)
    nk = SimpleNamespace(network_kwargs={"init_lora_weights": "pissa"})
    net = _linear_network(deferred_lora_init=True, network_config=nk)
    mod = LoRAModule("layer.ckpt", org, network=net, lora_dim=3)
    net.text_encoder_loras = []
    net.unet_loras = [mod]
    down_snapshot = mod.lora_down.weight.detach().clone()

    with patch("toolkit.lora_special.try_init_linear_lora_down_pissa") as mocked:
        finalize_deferred_lora_init(
            net,
            frozenset({f"{mod.lora_name}.lora_down.weight"}),
        )
        mocked.assert_not_called()

    assert mod._deferred_lora_init_pending is False
    assert torch.equal(down_snapshot, mod.lora_down.weight)
