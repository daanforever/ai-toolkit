"""
LoRA optimizer param-group split tests for z_image_diffsynth.
"""

from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.diffusion_models.z_image_diffsynth import lora as lora_mod
from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
    ZImageDiffSynthModel,
)
from toolkit.optimizers.adafactor import Adafactor


class DummyLoRA(torch.nn.Module):
    def __init__(self, lora_name: str):
        super().__init__()
        self.lora_name = lora_name
        self.weight = torch.nn.Parameter(torch.randn(1))


class DummyDoRALoRA(torch.nn.Module):
    def __init__(self, lora_name: str, has_magnitude: bool = False):
        super().__init__()
        self.lora_name = lora_name
        self.weight = torch.nn.Parameter(torch.randn(1))
        if has_magnitude:
            self.magnitude = torch.nn.Parameter(torch.randn(1))


def _make_model_stub() -> ZImageDiffSynthModel:
    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.print_and_status_update = lambda *_args, **_kwargs: None
    return model


@pytest.mark.parametrize(
    "lora_name,expected",
    [
        (
            "transformer$$_inner_dit$$layers$$0$$attention$$to_q",
            "layers_0",
        ),
        (
            "transformer$$_inner_dit$$layers$$29$$feed_forward$$w2",
            "layers_29",
        ),
        (
            "lora_unet__inner_dit_noise_refiner_1_attention_to_v",
            "noise_refiner_1",
        ),
        (
            "lora_transformer__inner_dit_context_refiner_0_feed_forward_w1",
            "context_refiner_0",
        ),
    ],
)
def test_parse_lora_block_key(lora_name, expected):
    assert lora_mod.parse_lora_block_key(lora_name) == expected


def test_group_loras_by_block_falls_back_to_other():
    loras = [
        DummyLoRA("transformer$$_inner_dit$$layers$$0$$attention$$to_q"),
        DummyLoRA("bad_lora_name"),
    ]

    grouped = lora_mod.group_loras_by_block(loras)

    assert list(grouped.keys()) == ["layers_0", "other"]
    assert len(grouped["layers_0"]) == 1
    assert len(grouped["other"]) == 1


def test_model_optimizer_groups_are_disjoint_per_block():
    loras = [
        DummyLoRA("transformer$$_inner_dit$$layers$$0$$attention$$to_q"),
        DummyLoRA("transformer$$_inner_dit$$layers$$0$$attention$$to_k"),
        DummyLoRA("transformer$$_inner_dit$$layers$$1$$feed_forward$$w1"),
        DummyLoRA("transformer$$_inner_dit$$noise_refiner$$0$$attention$$to_v"),
    ]
    model = _make_model_stub()
    network = SimpleNamespace(unet_loras=loras)

    param_groups = model.get_lora_optimizer_param_groups(
        network=network,
        unet_lr=1e-4,
        default_lr=1e-5,
    )

    assert param_groups is not None
    assert len(param_groups) == 3
    assert all(group["lr"] == pytest.approx(1e-4) for group in param_groups)

    param_to_expected_block = {}
    for lora in loras:
        block_key = lora_mod.parse_lora_block_key(lora.lora_name)
        assert block_key is not None
        for param in lora.parameters():
            param_to_expected_block[id(param)] = block_key

    seen_params = set()
    for group in param_groups:
        group_param_ids = [id(param) for param in group["params"]]
        assert len(group_param_ids) == len(set(group_param_ids))
        assert not any(param_id in seen_params for param_id in group_param_ids)
        seen_params.update(group_param_ids)

        expected_block = param_to_expected_block[group_param_ids[0]]
        for param_id in group_param_ids:
            assert param_to_expected_block[param_id] == expected_block

    assert len(seen_params) == len(param_to_expected_block)


def test_adafactor_accepts_30_block_groups():
    loras = [
        DummyLoRA(f"transformer$$_inner_dit$$layers$${idx}$$attention$$to_q")
        for idx in range(30)
    ]
    model = _make_model_stub()
    network = SimpleNamespace(unet_loras=loras)

    param_groups = model.get_lora_optimizer_param_groups(
        network=network,
        unet_lr=1e-4,
        default_lr=1e-5,
    )

    assert param_groups is not None
    assert len(param_groups) == 30

    optimizer = Adafactor(
        param_groups,
        lr=1e-4,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.0,
    )
    assert len(optimizer.param_groups) == 30


def test_model_optimizer_groups_dora_magnitude():
    loras = [
        DummyDoRALoRA("transformer$$_inner_dit$$layers$$0$$attention$$to_q", has_magnitude=True),
        DummyDoRALoRA("transformer$$_inner_dit$$layers$$0$$attention$$to_k", has_magnitude=True),
        DummyDoRALoRA("transformer$$_inner_dit$$layers$$1$$feed_forward$$w1", has_magnitude=False),
    ]
    model = _make_model_stub()
    network = SimpleNamespace(unet_loras=loras)

    param_groups = model.get_lora_optimizer_param_groups(
        network=network,
        unet_lr=1e-4,
        default_lr=1e-5,
    )

    print("PARAM GROUPS:", param_groups)

    assert param_groups is not None
    # We expect:
    # 1. layers_0 non-magnitude group
    # 2. layers_0 magnitude group
    # 3. layers_1 non-magnitude group
    # Total of 3 groups.
    assert len(param_groups) == 3

    # Check that magnitude group has is_magnitude=True and others do not.
    mag_groups = [group for group in param_groups if group.get("is_magnitude") is True]
    non_mag_groups = [group for group in param_groups if not group.get("is_magnitude")]

    assert len(mag_groups) == 1
    assert len(non_mag_groups) == 2

    # Verify parameters in magnitude group are indeed named "magnitude"
    mag_params = mag_groups[0]["params"]
    assert len(mag_params) == 2  # from layers_0 to_q and to_k

    # Verify non-magnitude groups contain the weights
    total_non_mag_params = sum(len(g["params"]) for g in non_mag_groups)
    assert total_non_mag_params == 3  # weight from to_q, to_k, and layers_1 w1


# ---------------------------------------------------------------------------
# PEFT-backed PeftNetwork tests
# ---------------------------------------------------------------------------

class Attention(torch.nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.to_q = torch.nn.Linear(d, d)
        self.to_k = torch.nn.Linear(d, d)
        self.to_v = torch.nn.Linear(d, d)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(d, d)])

    def forward(self, x):
        return self.to_out[0](self.to_v(x))


class FeedForward(torch.nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.w1 = torch.nn.Linear(d, d * 2)
        self.w2 = torch.nn.Linear(d * 2, d)
        self.w3 = torch.nn.Linear(d, d * 2)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) + self.w3(x))


class _BlockStub(torch.nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.attention = Attention(d)
        self.feed_forward = FeedForward(d)


class _InnerDiTStub(torch.nn.Module):
    def __init__(self, d=4, n_blocks=2, n_noise_refiner=1, n_context_refiner=1):
        super().__init__()
        self.layers = torch.nn.ModuleList([_BlockStub(d) for _ in range(n_blocks)])
        self.noise_refiner = torch.nn.ModuleList([_BlockStub(d) for _ in range(n_noise_refiner)])
        self.context_refiner = torch.nn.ModuleList([_BlockStub(d) for _ in range(n_context_refiner)])

    def forward(self, x):
        for blk in self.layers:
            x = blk.attention(x) + blk.feed_forward(x)
        return x


class _UnetWrapperStub(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self._inner_dit = dit

    def forward(self, *a, **k):
        return self._inner_dit(*a, **k)


class _StubBaseModel:
    """Minimal stand-in for ZImageDiffSynthModel for PeftNetwork construction."""
    arch = "zimage_diffsynth"
    target_lora_modules = ["Attention", "FeedForward"]

    def convert_lora_weights_before_save(self, sd):
        return {k.replace("transformer.", "diffusion_model."): v for k, v in sd.items()}

    def convert_lora_weights_before_load(self, sd):
        return {k.replace("diffusion_model.", "transformer."): v for k, v in sd.items()}

    def get_lora_optimizer_param_groups(self, network, unet_lr, default_lr):
        unet_loras = getattr(network, "unet_loras", None)
        if not unet_loras:
            return None
        grouped = lora_mod.group_loras_by_block(unet_loras)
        param_groups = []
        for _, block_loras in grouped.items():
            lora_params = []
            magnitude_params = []
            for lora in block_loras:
                for name, p in lora.named_parameters():
                    if "magnitude" in name:
                        magnitude_params.append(p)
                    else:
                        lora_params.append(p)
            if lora_params:
                g = {"params": lora_params}
                if unet_lr is not None:
                    g["lr"] = unet_lr
                param_groups.append(g)
            if magnitude_params:
                mg = {"params": magnitude_params, "is_magnitude": True}
                if unet_lr is not None:
                    mg["lr"] = unet_lr
                param_groups.append(mg)
        return param_groups or None


def _build_peft_network(network_type: str, n_blocks: int = 2):
    from toolkit.peft_network import PeftNetwork

    dit = _InnerDiTStub(d=4, n_blocks=n_blocks)
    wrapper = _UnetWrapperStub(dit)
    base = _StubBaseModel()
    net = PeftNetwork(
        text_encoder=None,
        unet=wrapper,
        multiplier=1.0,
        lora_dim=2,
        alpha=2.0,
        train_unet=True,
        train_text_encoder=False,
        network_type=network_type,
        base_model=base,
        target_lin_modules=base.target_lora_modules,
    )
    return net, base


def test_peft_param_groups_disjoint():
    """PeftNetwork.prepare_optimizer_params must return disjoint param groups
    grouped by DiT block, matching the LoRA path's contract."""
    net, _ = _build_peft_network("peft", n_blocks=3)

    groups = net.prepare_optimizer_params(text_encoder_lr=1e-4, unet_lr=1e-4, default_lr=1e-5)

    assert groups, "expected at least one param group"
    # 3 layers blocks + 1 noise_refiner + 1 context_refiner = 5 groups
    assert len(groups) == 5

    seen_ids = set()
    for group in groups:
        ids = [id(p) for p in group["params"]]
        # no duplicate within a group
        assert len(ids) == len(set(ids))
        # no overlap across groups
        assert not any(i in seen_ids for i in ids)
        seen_ids.update(ids)

    # every trainable lora param should appear exactly once
    all_trainable = []
    for adapter in net.unet_loras:
        for _, p in adapter.named_parameters():
            all_trainable.append(id(p))
    assert seen_ids == set(all_trainable)
    assert len(seen_ids) == len(all_trainable)


def test_peft_dora_param_groups_have_magnitude():
    """PeftNetwork with peft_dora must split magnitude params into a separate
    is_magnitude group per block."""
    net, _ = _build_peft_network("peft_dora", n_blocks=2)

    groups = net.prepare_optimizer_params(text_encoder_lr=1e-4, unet_lr=1e-4, default_lr=1e-5)

    mag_groups = [g for g in groups if g.get("is_magnitude") is True]
    non_mag_groups = [g for g in groups if not g.get("is_magnitude")]
    assert mag_groups, "expected at least one magnitude group for peft_dora"
    # 2 layers + 1 noise_refiner + 1 context_refiner = 4 magnitude groups
    assert len(mag_groups) == 4
    assert len(non_mag_groups) == 4

    # magnitude params must not appear in non-magnitude groups
    mag_ids = {id(p) for g in mag_groups for p in g["params"]}
    non_mag_ids = {id(p) for g in non_mag_groups for p in g["params"]}
    assert mag_ids.isdisjoint(non_mag_ids)


def test_peft_lora_name_matches_block_key_regex():
    """The lora_name produced by PeftNetwork must be parseable by the existing
    parse_lora_block_key regex (no extension needed)."""
    net, _ = _build_peft_network("peft", n_blocks=2)
    for adapter in net.unet_loras:
        block_key = lora_mod.parse_lora_block_key(adapter.lora_name)
        assert block_key is not None, f"failed to parse {adapter.lora_name!r}"
        assert block_key in {
            "layers_0", "layers_1",
            "noise_refiner_0", "context_refiner_0",
        }, f"unexpected block_key {block_key!r} for {adapter.lora_name!r}"


def test_peft_save_load_roundtrip():
    """Saved safetensors keys must round-trip back into a fresh PeftNetwork."""
    import os
    import tempfile

    from safetensors.torch import save_file, load_file

    net1, base = _build_peft_network("peft", n_blocks=2)
    sd = net1.get_state_dict(dtype=torch.float32)
    # Saved keys should be in DiffSynth convention (diffusion_model. prefix).
    sample_key = next(iter(sd.keys()))
    assert sample_key.startswith("diffusion_model."), sample_key
    assert ".lora_A.weight" in sample_key, sample_key

    tmp = tempfile.mktemp(suffix=".safetensors")
    try:
        save_file(sd, tmp)
        loaded = load_file(tmp)
        loaded = base.convert_lora_weights_before_load(loaded)

        net2, _ = _build_peft_network("peft", n_blocks=2)
        net2.load_weights(loaded)

        # Compare a couple of lora_A weights
        a_w = net1.peft_model.state_dict()["base_model.model._inner_dit.layers.0.attention.to_q.lora_A.default.weight"]
        b_w = net2.peft_model.state_dict()["base_model.model._inner_dit.layers.0.attention.to_q.lora_A.default.weight"]
        assert torch.allclose(a_w, b_w), "lora_A weights differ after round-trip"
    finally:
        try:
            os.remove(tmp)
        except PermissionError:
            pass


def test_peft_share_parameters_with():
    """share_parameters_with must make the sampling network's lora params the
    same object references as the main network's."""
    net1, _ = _build_peft_network("peft", n_blocks=2)
    net2, _ = _build_peft_network("peft", n_blocks=2)

    # Access the underlying nn.Parameter objects directly (state_dict() would
    # return clones and obscure the sharing check).
    layer1 = net1.unet_loras[0].layer
    layer2 = net2.unet_loras[0].layer
    a_w = layer1.lora_A["default"].weight
    b_w_before = layer2.lora_A["default"].weight
    assert a_w is not b_w_before

    net2.share_parameters_with(net1)

    b_w_after = layer2.lora_A["default"].weight
    assert a_w is b_w_after, "lora_A weight not shared by reference"

    # Mutating the main network's weight should be visible in the sampling network.
    with torch.no_grad():
        a_w.add_(1.0)
    assert torch.allclose(a_w, b_w_after)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
