"""Heatmap metrics for kohya-style LoRA safetensors files."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from toolkit.lora_heatmap import (
    METRIC_BALANCE,
    METRIC_EFFECTIVE_RANK,
    METRIC_FROBENIUS,
    METRIC_NAMES,
    METRIC_STABLE_RANK,
    UnsupportedLoRAFormat,
    compute_lora_heatmap,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "toolkit", "lora_heatmap.py")
_FLOWER = os.path.join(_REPO, "temp", "flower.safetensors")


def _save(path, tensors, metadata=None):
    save_file(tensors, path, metadata=metadata)


def _pair(prefix, down, up, alpha=None):
    tensors = {
        f"{prefix}.lora_down.weight": down,
        f"{prefix}.lora_up.weight": up,
    }
    if alpha is not None:
        tensors[f"{prefix}.alpha"] = alpha
    return tensors


def _brute_frobenius(down, up, scale):
    matrix_down = down.float()
    matrix_up = up.float()
    total = torch.zeros((), dtype=torch.float32)
    chunk = 256
    for start in range(0, matrix_down.shape[1], chunk):
        piece = matrix_up @ matrix_down[:, start : start + chunk]
        total = total + torch.sum(piece * piece)
    return float(scale * torch.sqrt(total))


def _brute_ranks(down, up):
    product = up.float() @ down.float()
    sigmas = torch.linalg.svdvals(product)
    sigma_max = sigmas.max()
    stable = float(torch.sum(sigmas * sigmas) / (sigma_max * sigma_max))
    probs = sigmas / torch.sum(sigmas)
    probs = probs[probs > 0]
    effective = float(torch.exp(-torch.sum(probs * torch.log(probs))))
    return stable, effective


def test_metrics_match_direct_product(tmp_path):
    generator = torch.Generator().manual_seed(7)
    down = torch.randn(4, 8, generator=generator, dtype=torch.float32)
    up = torch.randn(6, 4, generator=generator, dtype=torch.float32)
    alpha = torch.tensor(8.0)
    path = str(tmp_path / "lora.safetensors")
    _save(
        path,
        _pair("lora_unet__inner_dit_layers_0_attention_to_q", down, up, alpha),
        metadata={"training_info": json.dumps({"step": 12, "epoch": 3})},
    )

    payload = compute_lora_heatmap(path)
    assert payload["rows"] == ["layers_0"]
    assert payload["cols"] == ["attention_to_q"]
    assert payload["file"]["step"] == 12
    assert payload["file"]["epoch"] == 3

    scale = 8.0 / 4.0
    assert payload["metrics"][METRIC_FROBENIUS][0][0] == pytest.approx(_brute_frobenius(down, up, scale))
    stable, effective = _brute_ranks(down, up)
    assert payload["metrics"][METRIC_STABLE_RANK][0][0] == pytest.approx(stable, rel=1e-4, abs=1e-4)
    assert payload["metrics"][METRIC_EFFECTIVE_RANK][0][0] == pytest.approx(effective, rel=1e-4, abs=1e-4)
    assert 1.0 <= payload["metrics"][METRIC_STABLE_RANK][0][0] <= 4.0 + 1e-4
    assert 1.0 <= payload["metrics"][METRIC_EFFECTIVE_RANK][0][0] <= 4.0 + 1e-4
    assert payload["metrics"][METRIC_BALANCE][0][0] == pytest.approx(
        float(torch.linalg.norm(up.float()) / torch.linalg.norm(down.float()))
    )


def test_bf16_matches_direct_product(tmp_path):
    generator = torch.Generator().manual_seed(3)
    down = torch.randn(4, 8, generator=generator).to(torch.bfloat16)
    up = torch.randn(5, 4, generator=generator).to(torch.bfloat16)
    path = str(tmp_path / "bf16.safetensors")
    _save(path, _pair("lora_unet__blocks_1_to_v", down, up, torch.tensor(4.0, dtype=torch.bfloat16)))
    payload = compute_lora_heatmap(path)
    scale = 1.0
    assert payload["metrics"][METRIC_FROBENIUS][0][0] == pytest.approx(
        _brute_frobenius(down, up, scale), rel=1e-5, abs=1e-5
    )


def test_rank_smaller_than_factors_uses_product_svd(tmp_path):
    generator = torch.Generator().manual_seed(11)
    down = torch.randn(4, 3, generator=generator)
    up = torch.randn(2, 4, generator=generator)
    path = str(tmp_path / "thin.safetensors")
    _save(path, _pair("lora_unet__layers_0_mlp", down, up, torch.tensor(4.0)))
    payload = compute_lora_heatmap(path)
    stable, effective = _brute_ranks(down, up)
    assert payload["metrics"][METRIC_STABLE_RANK][0][0] == pytest.approx(stable, rel=1e-4, abs=1e-4)
    assert payload["metrics"][METRIC_EFFECTIVE_RANK][0][0] == pytest.approx(effective, rel=1e-4, abs=1e-4)


def test_zero_up_nulls_ratio_and_ranks(tmp_path):
    down = torch.randn(4, 6)
    up = torch.zeros(6, 4)
    path = str(tmp_path / "zero.safetensors")
    _save(path, _pair("lora_unet__layers_0_to_q", down, up, torch.tensor(4.0)))
    payload = compute_lora_heatmap(path)
    assert payload["metrics"][METRIC_FROBENIUS][0][0] == pytest.approx(0.0, abs=1e-8)
    assert payload["metrics"][METRIC_BALANCE][0][0] is None
    assert payload["metrics"][METRIC_STABLE_RANK][0][0] is None
    assert payload["metrics"][METRIC_EFFECTIVE_RANK][0][0] is None


def test_missing_alpha_uses_unit_scale(tmp_path):
    generator = torch.Generator().manual_seed(5)
    down = torch.randn(2, 3, generator=generator)
    up = torch.randn(3, 2, generator=generator)
    path = str(tmp_path / "noalpha.safetensors")
    _save(path, _pair("lora_unet__layers_0_to_k", down, up, alpha=None))
    payload = compute_lora_heatmap(path)
    assert payload["metrics"][METRIC_FROBENIUS][0][0] == pytest.approx(_brute_frobenius(down, up, 1.0))


def test_zero_alpha_uses_unit_scale(tmp_path):
    down = torch.eye(2)
    up = torch.eye(2)
    path = str(tmp_path / "zeroalpha.safetensors")
    _save(path, _pair("lora_unet__layers_0_to_q", down, up, torch.tensor(0.0)))
    payload = compute_lora_heatmap(path)
    assert payload["metrics"][METRIC_FROBENIUS][0][0] == pytest.approx(_brute_frobenius(down, up, 1.0))


def test_key_without_layer_index_is_one_row(tmp_path):
    down = torch.randn(2, 3)
    up = torch.randn(3, 2)
    path = str(tmp_path / "nolayer.safetensors")
    _save(path, _pair("lora_te_text_model", down, up, torch.tensor(2.0)))
    payload = compute_lora_heatmap(path)
    assert payload["rows"] == ["other"]
    assert payload["cols"] == ["lora_te_text_model"]


def test_missing_cells_and_block_names(tmp_path):
    generator = torch.Generator().manual_seed(9)
    tensors = {}
    tensors.update(
        _pair(
            "lora_unet__inner_dit_layers_0_attention_to_q",
            torch.randn(2, 4, generator=generator),
            torch.randn(4, 2, generator=generator),
            torch.tensor(2.0),
        )
    )
    tensors.update(
        _pair(
            "lora_unet__inner_dit_layers_1_attention_to_k",
            torch.randn(2, 4, generator=generator),
            torch.randn(4, 2, generator=generator),
            torch.tensor(2.0),
        )
    )
    tensors.update(
        _pair(
            "lora_unet__inner_dit_noise_refiner_1_attention_to_v",
            torch.randn(2, 4, generator=generator),
            torch.randn(4, 2, generator=generator),
            torch.tensor(2.0),
        )
    )
    tensors.update(
        _pair(
            "diffusion_model.layers.2.attention.to_out.0",
            torch.randn(2, 4, generator=generator),
            torch.randn(4, 2, generator=generator),
            torch.tensor(2.0),
        )
    )
    path = str(tmp_path / "grid.safetensors")
    _save(path, tensors)
    payload = compute_lora_heatmap(path)
    assert payload["rows"] == ["layers_0", "layers_1", "layers_2", "noise_refiner_1"]
    assert payload["cols"] == ["attention_to_k", "attention_to_out_0", "attention_to_q", "attention_to_v"]
    frobenius = payload["metrics"][METRIC_FROBENIUS]
    assert frobenius[0][2] is not None
    assert frobenius[0][0] is None
    assert frobenius[1][0] is not None
    assert frobenius[2][1] is not None
    assert frobenius[3][3] is not None


def test_peft_keys_are_rejected(tmp_path):
    path = str(tmp_path / "peft.safetensors")
    _save(
        path,
        {
            "blocks.0.attn.lora_A.weight": torch.randn(4, 8),
            "blocks.0.attn.lora_B.weight": torch.randn(8, 4),
        },
    )
    with pytest.raises(UnsupportedLoRAFormat, match="lora_A"):
        compute_lora_heatmap(path)


def test_lokr_keys_are_rejected(tmp_path):
    path = str(tmp_path / "lokr.safetensors")
    _save(path, {"lycoris_transformer_blocks_0_attn.lokr_w1": torch.randn(4, 4)})
    with pytest.raises(UnsupportedLoRAFormat, match="LoKr"):
        compute_lora_heatmap(path)


def test_conv_weights_are_rejected(tmp_path):
    path = str(tmp_path / "conv.safetensors")
    _save(
        path,
        _pair(
            "lora_unet__layers_0_conv",
            torch.randn(4, 3, 3, 3),
            torch.randn(8, 4, 1, 1),
            torch.tensor(4.0),
        ),
    )
    with pytest.raises(UnsupportedLoRAFormat, match="Conv"):
        compute_lora_heatmap(path)


def test_cli_prints_json_and_errors(tmp_path):
    path = str(tmp_path / "lora.safetensors")
    _save(path, _pair("lora_unet__layers_0_to_q", torch.eye(2), torch.eye(2), torch.tensor(2.0)))
    ok = subprocess.run([sys.executable, _SCRIPT, path], capture_output=True, text=True, check=False)
    assert ok.returncode == 0, ok.stderr
    payload = json.loads(ok.stdout)
    assert payload["rows"] == ["layers_0"]
    assert set(payload["metrics"]) == set(METRIC_NAMES)

    bad = subprocess.run(
        [sys.executable, _SCRIPT, str(tmp_path / "missing.safetensors")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1
    assert bad.stderr.strip() != ""

    usage = subprocess.run([sys.executable, _SCRIPT], capture_output=True, text=True, check=False)
    assert usage.returncode == 2

    corrupt = tmp_path / "corrupt.safetensors"
    corrupt.write_bytes(b"not a safetensors file")
    failed = subprocess.run([sys.executable, _SCRIPT, str(corrupt)], capture_output=True, text=True, check=False)
    assert failed.returncode == 1
    assert failed.stderr.strip() != ""


@pytest.mark.skipif(not os.path.isfile(_FLOWER), reason="temp/flower.safetensors is not present")
def test_flower_reference_file():
    payload = compute_lora_heatmap(_FLOWER)
    assert len(payload["rows"]) == 30
    assert payload["rows"][0] == "layers_0"
    assert payload["rows"][-1] == "layers_29"
    assert len(payload["cols"]) == 7
    assert payload["file"]["step"] == 10
    for name in METRIC_NAMES:
        matrix = payload["metrics"][name]
        assert len(matrix) == 30
        assert all(len(row) == 7 for row in matrix)
        for row in matrix:
            for value in row:
                assert value is not None and math.isfinite(value)
                if name in (METRIC_STABLE_RANK, METRIC_EFFECTIVE_RANK):
                    assert 1.0 <= value <= 16.0 + 1e-3

    from safetensors import safe_open

    checked = 0
    with safe_open(_FLOWER, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.endswith(".lora_down.weight"):
                continue
            module = key[: -len(".lora_down.weight")]
            down = handle.get_tensor(key)
            up = handle.get_tensor(f"{module}.lora_up.weight")
            alpha = handle.get_tensor(f"{module}.alpha")
            rank = int(down.shape[0])
            scale = float(alpha.float().reshape(())) / float(rank)
            layer_index, col_name = module.split("layers_", 1)[1].split("_", 1)
            row_name = f"layers_{layer_index}"
            i = payload["rows"].index(row_name)
            j = payload["cols"].index(col_name)
            assert payload["metrics"][METRIC_FROBENIUS][i][j] == pytest.approx(
                _brute_frobenius(down, up, scale), rel=1e-4, abs=1e-5
            )
            assert rank == 16
            checked += 1
            if checked == 3:
                break
    assert checked == 3
