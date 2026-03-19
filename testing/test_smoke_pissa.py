"""Smoke checks for PiSSA LoRA-down initialization helper."""

import torch

from toolkit.lora_utils.pissa import compute_pissa_linear_lora_down


def test_pissa_returns_expected_shape_and_finite():
    torch.manual_seed(0)
    out_f, in_f, rank = 8, 6, 3
    W = torch.randn(out_f, in_f)
    down, fail = compute_pissa_linear_lora_down(W, rank)
    assert down is not None
    assert fail is None
    assert down.shape == (rank, in_f)
    assert down.dtype == torch.float32
    assert torch.isfinite(down).all()


def test_pissa_invalid_rank_returns_none():
    W = torch.randn(4, 5)
    d0, r0 = compute_pissa_linear_lora_down(W, 0)
    assert d0 is None and r0 is not None
    assert r0.startswith("rank<=0")
    d99, r99 = compute_pissa_linear_lora_down(W, 99)
    assert d99 is None and r99 is not None
    assert r99.startswith("rank>min(out_f,in_f):")


def test_pissa_fails_when_requested_rank_exceeds_weight_matrix_min_extent():
    """Narrow Linear (e.g. noise_refiner attention) with global LoRA rank larger than min(out,in)."""
    out_f, in_f, rank = 64, 64, 128
    W = torch.randn(out_f, in_f)
    down, reason = compute_pissa_linear_lora_down(W, rank)
    assert down is None
    assert reason == (
        f"rank>min(out_f,in_f): rank={rank}, out_f={out_f}, in_f={in_f}, min={min(out_f, in_f)}"
    )
