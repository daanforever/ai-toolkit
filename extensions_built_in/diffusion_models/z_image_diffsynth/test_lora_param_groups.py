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


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
