"""Smoke checks for PiSSA LoRA-down initialization helper."""

import torch

from toolkit.lora_utils.pissa import compute_pissa_linear_lora_down


def test_pissa_returns_expected_shape_and_finite():
    torch.manual_seed(0)
    out_f, in_f, rank = 8, 6, 3
    W = torch.randn(out_f, in_f)
    down = compute_pissa_linear_lora_down(W, rank)
    assert down is not None
    assert down.shape == (rank, in_f)
    assert down.dtype == torch.float32
    assert torch.isfinite(down).all()


def test_pissa_invalid_rank_returns_none():
    W = torch.randn(4, 5)
    assert compute_pissa_linear_lora_down(W, 0) is None
    assert compute_pissa_linear_lora_down(W, 99) is None
