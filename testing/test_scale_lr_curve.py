"""Scale-LR-by-index curve from kohya-style LoRA safetensors files."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from toolkit.lora_block_index import block_index, parse_lora_row_col
from toolkit.optimizers.adafactor import Adafactor
from toolkit.scale_lr_curve import compute_scale_lr_curve, main

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "toolkit", "scale_lr_curve.py")
_FLOWER = os.path.join(_REPO, "temp", "flower.safetensors")


def _save(path, tensors, metadata=None):
    save_file(tensors, path, metadata=metadata)


def _pair(prefix):
    return {
        f"{prefix}.lora_down.weight": torch.zeros(1, 1),
        f"{prefix}.lora_up.weight": torch.zeros(1, 1),
    }


def _layers(count):
    tensors = {}
    for index in range(count):
        tensors.update(_pair(f"lora_unet__layers_{index}_to_q"))
    return tensors


def test_parse_keeps_longest_block_name():
    assert parse_lora_row_col("diffusion_model.double_blocks.2.attn") == (
        "double_blocks_2",
        "attn",
    )
    assert parse_lora_row_col("lora_unet__blocks_1_to_v") == ("blocks_1", "to_v")
    assert parse_lora_row_col("lora_te_text_model") == ("other", "lora_te_text_model")
    assert block_index("layers_12") == 12
    assert block_index("other") is None


def test_weights_match_optimizer(tmp_path):
    path = str(tmp_path / "layers.safetensors")
    _save(
        path,
        _layers(30),
        metadata={"training_info": json.dumps({"step": 10, "epoch": 1})},
    )
    payload = compute_scale_lr_curve(path, 15.0, 0.25)
    expected = Adafactor._compute_gaussian_lr_weights(30, 15.0, 0.25)
    assert payload["n"] == 30
    assert payload["indices"] == list(range(30))
    assert payload["file"]["step"] == 10
    assert payload["file"]["filename"] == "layers.safetensors"
    for index, value in enumerate(payload["weights"]):
        assert value == pytest.approx(float(expected[index].item()))


def test_mask_matching_nothing_is_all_ones(tmp_path):
    path = str(tmp_path / "layers.safetensors")
    _save(path, _layers(4))
    payload = compute_scale_lr_curve(path, 1.0, 0.5, ["zzz"])
    assert payload["weights"] == [1.0, 1.0, 1.0, 1.0]


def test_mask_matches_row_id(tmp_path):
    path = str(tmp_path / "mixed.safetensors")
    tensors = {}
    tensors.update(_pair("lora_unet__layers_0_to_q"))
    tensors.update(_pair("lora_unet__layers_1_to_q"))
    tensors.update(_pair("lora_unet__blocks_1_to_v"))
    _save(path, tensors)
    payload = compute_scale_lr_curve(path, 0.0, 0.5, ["blocks"])
    expected = Adafactor._compute_gaussian_lr_weights(2, 0.0, 0.5)
    assert payload["weights"][0] == 1.0
    assert payload["weights"][1] == pytest.approx(float(expected[1].item()))


def test_gap_without_mask_stays_on_the_gaussian(tmp_path):
    path = str(tmp_path / "gap.safetensors")
    tensors = {}
    tensors.update(_pair("lora_unet__layers_0_to_q"))
    tensors.update(_pair("lora_unet__layers_2_to_q"))
    _save(path, tensors)
    payload = compute_scale_lr_curve(path, 1.0, 0.5)
    expected = Adafactor._compute_gaussian_lr_weights(3, 1.0, 0.5)
    assert payload["n"] == 3
    for index, value in enumerate(payload["weights"]):
        assert value == pytest.approx(float(expected[index].item()))


def test_mean_outside_the_range_peaks_at_the_edge(tmp_path):
    path = str(tmp_path / "layers.safetensors")
    _save(path, _layers(5))
    payload = compute_scale_lr_curve(path, 100.0, 0.2)
    weights = payload["weights"]
    assert weights[-1] == max(weights)
    assert weights[-1] == pytest.approx(1.0)


def test_single_index_is_rejected(tmp_path):
    path = str(tmp_path / "one.safetensors")
    _save(path, _pair("lora_unet__layers_0_to_q"))
    with pytest.raises(ValueError, match="max_index > 0"):
        compute_scale_lr_curve(path, 0.0, 0.5)


def test_no_layer_index_is_rejected(tmp_path):
    path = str(tmp_path / "other.safetensors")
    _save(path, _pair("lora_te_text_model"))
    with pytest.raises(ValueError, match="No layer indices"):
        compute_scale_lr_curve(path, 1.0, 0.5)


def test_std_must_be_positive(tmp_path):
    path = str(tmp_path / "layers.safetensors")
    _save(path, _layers(3))
    with pytest.raises(ValueError, match="scale_lr_std"):
        compute_scale_lr_curve(path, 1.0, 0.0)
    with pytest.raises(ValueError, match="scale_lr_std"):
        compute_scale_lr_curve(path, 1.0, float("nan"))
    with pytest.raises(ValueError, match="scale_lr_mean"):
        compute_scale_lr_curve(path, float("nan"), 0.5)


def test_cli_prints_json_and_usage(tmp_path):
    path = str(tmp_path / "layers.safetensors")
    _save(path, _layers(3), metadata={"training_info": json.dumps({"step": 4, "epoch": 1})})
    ok = subprocess.run(
        [sys.executable, _SCRIPT, path, "--mean", "1", "--std", "0.5"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    payload = json.loads(ok.stdout)
    assert payload["n"] == 3
    assert payload["file"]["step"] == 4
    assert len(payload["weights"]) == 3

    usage = main([])
    assert usage == 2

    bad = subprocess.run(
        [sys.executable, _SCRIPT, path, "--mean", "1", "--std", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1
    assert "scale_lr_std" in bad.stderr


@pytest.mark.skipif(not os.path.isfile(_FLOWER), reason="temp/flower.safetensors is not present")
def test_flower_matches_optimizer():
    payload = compute_scale_lr_curve(_FLOWER, 15.0, 0.25)
    expected = Adafactor._compute_gaussian_lr_weights(payload["n"], 15.0, 0.25)
    assert payload["n"] == 30
    assert payload["file"]["step"] == 10
    for index, value in enumerate(payload["weights"]):
        assert value == pytest.approx(float(expected[index].item()))
