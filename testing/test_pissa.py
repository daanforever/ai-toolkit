"""Tests for PiSSA LoRA-down init, including quantized base weights.

torchao's ``AffineQuantizedTensor`` hooks ``aten.mm`` assuming the second operand is the
quantized weight; PiSSA uses ``W @ Q`` with a float ``Q``. Materializing ``W`` to a plain
float matrix before matmul avoids ``AttributeError: 'Tensor' object has no attribute
'_quantized_linear_op'``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final

import pytest
import torch

from toolkit.lora_utils.pissa import (
    _materialize_float_weight_matrix,
    compute_pissa_linear_lora_down,
    try_init_linear_lora_down_pissa,
)


def _check_torchao() -> bool:
    try:
        import torchao  # noqa: F401
    except ImportError:
        return False
    return True


def _check_optimum_quanto() -> bool:
    try:
        import optimum.quanto  # noqa: F401
    except ImportError:
        return False
    return True


def _torchao_quant_imports():
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    return quantize_, Int8WeightOnlyConfig


_REPO_ROOT: Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_QUANTO_PISSA_SUBPROC_TIMEOUT_SEC: Final[float] = 60.0


def test_materialize_plain_weight_is_float32_2d():
    w = torch.randn(10, 7, dtype=torch.bfloat16)
    m = _materialize_float_weight_matrix(w)
    assert m.shape == (10, 7)
    assert m.dtype == torch.float32
    assert type(m) is torch.Tensor


def test_compute_pissa_plain_weight_shape_and_finite():
    torch.manual_seed(0)
    out_f, in_f, rank = 16, 12, 4
    w = torch.randn(out_f, in_f)
    down, err = compute_pissa_linear_lora_down(w, rank)
    assert err is None
    assert down is not None
    assert down.shape == (rank, in_f)
    assert torch.isfinite(down).all()


def test_compute_pissa_rank_1():
    torch.manual_seed(3)
    w = torch.randn(32, 24)
    down, err = compute_pissa_linear_lora_down(w, rank=1)
    assert err is None
    assert down is not None
    assert down.shape == (1, 24)
    assert torch.isfinite(down).all()


def test_compute_pissa_rank_128_square_128_full_min_extent():
    """LoRA rank 128 with W square 128×128 (rank == min(out,in))."""
    torch.manual_seed(4)
    rank = 128
    w = torch.randn(rank, rank)
    down, err = compute_pissa_linear_lora_down(w, rank=rank)
    assert err is None, err
    assert down is not None
    assert down.shape == (rank, rank)
    assert torch.isfinite(down).all()


def test_compute_pissa_rank_128_rectangular_min_dim_128():
    """Wide layer: min(out,in)==128, global rank 128 (typical cap scenario)."""
    torch.manual_seed(5)
    out_f, in_f, rank = 256, 128, 128
    w = torch.randn(out_f, in_f)
    down, err = compute_pissa_linear_lora_down(w, rank=rank)
    assert err is None, err
    assert down is not None
    assert down.shape == (rank, in_f)
    assert torch.isfinite(down).all()


def test_compute_pissa_rank_128_exceeds_min_dim_fails():
    """Same nominal rank as training (128) but layer smaller → PiSSA rejects, no tensor."""
    torch.manual_seed(6)
    out_f, in_f, rank = 64, 64, 128
    w = torch.randn(out_f, in_f)
    down, err = compute_pissa_linear_lora_down(w, rank=rank)
    assert down is None
    assert err is not None
    assert err.startswith("rank>min(out_f,in_f):")


def test_try_init_pissa_rank_128_fills_lora_down():
    """``try_init_linear_lora_down_pissa``: rank 128 when min(out,in) allows (alpha only scales forward)."""
    torch.manual_seed(7)
    in_dim, out_dim, rank = 128, 128, 128
    org = torch.nn.Linear(in_dim, out_dim, bias=False)
    lora_down = torch.nn.Linear(in_dim, rank, bias=False)
    ok = try_init_linear_lora_down_pissa(
        init_lora_weights="pissa",
        org_module_class_name="Linear",
        full_rank=False,
        org_weight=org.weight,
        lora_down=lora_down,
        lora_dim=rank,
        in_dim=in_dim,
        out_dim=out_dim,
        network=None,
        lora_name="boundary_rank128",
    )
    assert ok is True
    assert torch.isfinite(lora_down.weight).all()
    assert lora_down.weight.shape == (rank, in_dim)


def test_try_init_pissa_global_rank_128_skipped_on_64x64_layer():
    """Training rank 128 with a 64×64 Linear: PiSSA branch skipped (lora_dim > min(out,in))."""
    torch.manual_seed(8)
    in_dim = out_dim = 64
    org = torch.nn.Linear(in_dim, out_dim, bias=False)
    lora_down = torch.nn.Linear(in_dim, 128, bias=False)
    ok = try_init_linear_lora_down_pissa(
        init_lora_weights="pissa",
        org_module_class_name="Linear",
        full_rank=False,
        org_weight=org.weight,
        lora_down=lora_down,
        lora_dim=128,
        in_dim=in_dim,
        out_dim=out_dim,
        network=None,
        lora_name="narrow_layer",
    )
    assert ok is False


@pytest.mark.skipif(
    not _check_torchao(),
    reason="torchao not installed",
)
def test_materialize_torchao_affine_quantized_is_plain_float():
    quantize_, Int8WeightOnlyConfig = _torchao_quant_imports()
    lin = torch.nn.Linear(12, 20, bias=False)
    quantize_(lin, Int8WeightOnlyConfig())
    w = lin.weight
    assert type(w).__name__ == "AffineQuantizedTensor"
    m = _materialize_float_weight_matrix(w)
    assert m.shape == (20, 12)
    assert m.dtype == torch.float32
    assert type(m) is torch.Tensor


@pytest.mark.skipif(
    not _check_torchao(),
    reason="torchao not installed",
)
def test_compute_pissa_torchao_int8_weight_only_linear():
    """Regression: PiSSA must not fail on torchao-quantized Linear weights."""
    quantize_, Int8WeightOnlyConfig = _torchao_quant_imports()
    lin = torch.nn.Linear(8, 16, bias=False)
    torch.manual_seed(1)
    quantize_(lin, Int8WeightOnlyConfig())
    down, err = compute_pissa_linear_lora_down(lin.weight, rank=4)
    assert err is None, err
    assert down is not None
    assert down.shape == (4, 8)
    assert torch.isfinite(down).all()


@pytest.mark.skipif(
    not _check_optimum_quanto(),
    reason="optimum.quanto not installed",
)
def test_compute_pissa_quanto_qbytes_weight():
    """Run in a subprocess so ``subprocess.run(..., timeout=...)`` can stop a hang (stdlib)."""
    env = os.environ.copy()
    sep = os.pathsep
    env["PYTHONPATH"] = _REPO_ROOT + sep + env.get("PYTHONPATH", "")

    script = r"""
import torch
from optimum.quanto import qint8
from optimum.quanto.tensor.qweight import quantize_weight
from toolkit.lora_utils.pissa import compute_pissa_linear_lora_down

torch.manual_seed(2)
w = torch.randn(16, 8)
qw = quantize_weight(w, qint8, axis=0)
down, err = compute_pissa_linear_lora_down(qw, rank=4)
assert err is None, err
assert down is not None
assert down.shape == (4, 8)
assert torch.isfinite(down).all()
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=_QUANTO_PISSA_SUBPROC_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"test_compute_pissa_quanto_qbytes_weight exceeded "
            f"{_QUANTO_PISSA_SUBPROC_TIMEOUT_SEC}s (subprocess timeout)"
        ) from e

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
