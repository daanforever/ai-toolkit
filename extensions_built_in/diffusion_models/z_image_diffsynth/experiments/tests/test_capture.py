"""capture Delta metrics."""

from __future__ import annotations

import torch

from ._loaders import load_capture


def test_update_rms_and_cosine_scaled():
    cap = load_capture()
    warm = {
        "transformer._inner_dit.layers.0.attention.to_q.lora_down.weight": torch.zeros(4, 8),
        "transformer._inner_dit.layers.0.attention.to_q.lora_up.weight": torch.zeros(8, 4),
    }
    base = {
        k: v + 0.1 for k, v in warm.items()
    }
    # Make non-zero structured delta
    base = {
        "transformer._inner_dit.layers.0.attention.to_q.lora_down.weight": torch.ones(4, 8) * 0.1,
        "transformer._inner_dit.layers.0.attention.to_q.lora_up.weight": torch.ones(8, 4) * 0.05,
    }
    other = {k: warm[k] + 4.0 * (base[k] - warm[k]) for k in base}
    # warm is zeros in this setup
    warm = {k: torch.zeros_like(v) for k, v in base.items()}
    other = {k: 4.0 * base[k] for k in base}

    d_base = cap.delta_tensors(warm, base)
    d_other = cap.delta_tensors(warm, other)
    assert abs(cap.cosine_delta(d_base, d_other) - 1.0) < 1e-5
    ratio = cap.update_rms(d_other) / cap.update_rms(d_base)
    assert abs(ratio - 4.0) < 1e-5

    summ = cap.summarize_delta_pair(warm, base, other)
    assert abs(summ["cosine"] - 1.0) < 1e-5
    assert abs(summ["ratio"] - 4.0) < 1e-5


def test_identical_cosine_one():
    cap = load_capture()
    warm = {"a.lora_down.weight": torch.zeros(2, 3)}
    after = {"a.lora_down.weight": torch.ones(2, 3)}
    d = cap.delta_tensors(warm, after)
    assert abs(cap.cosine_delta(d, d) - 1.0) < 1e-5
    assert abs(cap.update_rms(d) - 1.0) < 1e-5


def test_orthogonal_cosine_near_zero():
    cap = load_capture()
    a = {"x.lora_down.weight": torch.tensor([[1.0, 0.0]])}
    b = {"x.lora_down.weight": torch.tensor([[0.0, 1.0]])}
    assert abs(cap.cosine_delta(a, b)) < 1e-5


def test_down_up_split():
    cap = load_capture()
    warm = {
        "m.lora_down.weight": torch.zeros(2, 4),
        "m.lora_up.weight": torch.zeros(4, 2),
    }
    after = {
        "m.lora_down.weight": torch.ones(2, 4),
        "m.lora_up.weight": torch.ones(4, 2) * 2,
    }
    d = cap.delta_tensors(warm, after)
    down, up = cap.split_down_up(d)
    assert len(down) == 1 and len(up) == 1
    assert abs(cap.update_rms(down) - 1.0) < 1e-5
    assert abs(cap.update_rms(up) - 2.0) < 1e-5


def test_subtract_deltas():
    cap = load_capture()
    a = {"x.lora_down.weight": torch.tensor([[4.0, 0.0], [0.0, 4.0]])}
    b = {"x.lora_down.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    diff = cap.subtract_deltas(a, b)
    expected = {"x.lora_down.weight": torch.tensor([[3.0, 0.0], [0.0, 3.0]])}
    assert abs(cap.update_rms(diff) - cap.update_rms(expected)) < 1e-5
    assert abs(cap.cosine_delta(diff, expected) - 1.0) < 1e-5
    # a − b vs raw a−b element check
    assert torch.allclose(diff["x.lora_down.weight"], a["x.lora_down.weight"] - b["x.lora_down.weight"])
