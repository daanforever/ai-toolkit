"""dir_consistency uses a fixed fp32 coordinate sketch, not a full-size EMA."""
import math

import pytest
import torch
import torch.nn as nn

from toolkit.optimizers.adafactor import Adafactor


def _opt(params, **kwargs):
    defaults = dict(
        lr=1e-4,
        beta1=None,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        stochastic_accumulation=False,
        factored=False,
    )
    defaults.update(kwargs)
    return Adafactor(params, **defaults)


def _ref_ema(previous, sample, beta2t):
    return previous.mul(beta2t).add(sample, alpha=(1.0 - beta2t))


def test_small_direction_ema_matches_full_formula():
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(8, 16))
    opt = _opt([p])
    real = torch.nn.functional.cosine_similarity
    captured = {}

    def wrapped(x1, x2, dim=0, eps=1e-8):
        captured["sample"] = x1.detach().reshape(-1).clone()
        captured["ema"] = x2.detach().reshape(-1).clone()
        return real(x1, x2, dim=dim, eps=eps)

    torch.nn.functional.cosine_similarity = wrapped
    try:
        for _ in range(5):
            p.grad = torch.randn_like(p)
            opt.step()
            beta2t = opt.state[p]["beta2t"]
            expected = _ref_ema(captured["ema"], captured["sample"], beta2t)
            got = opt.state[p]["beta2_direction_ema"].detach().reshape(-1)
            assert got.dtype == torch.float32
            assert got.numel() == p.numel()
            assert torch.equal(got, expected)
            dc = float(opt.state[p]["dir_consistency"])
            assert math.isfinite(dc)
            assert -1.0 <= dc <= 1.0
    finally:
        torch.nn.functional.cosine_similarity = real


def test_sketch_does_not_change_parameter_update():
    torch.manual_seed(1)
    p1 = nn.Parameter(torch.randn(8, 16))
    p2 = nn.Parameter(p1.detach().clone())
    opt1 = _opt([p1], stochastic_rounding=False)
    opt2 = _opt([p2], stochastic_rounding=False)
    grad = torch.randn_like(p1)
    p1.grad = grad.clone()
    p2.grad = grad.clone()
    opt1.step()
    opt2.step()
    opt2.state[p2]["beta2_direction_ema"] = torch.randn_like(
        opt2.state[p2]["beta2_direction_ema"]
    )
    grad = torch.randn_like(p1)
    p1.grad = grad.clone()
    p2.grad = grad.clone()
    opt1.step()
    opt2.step()
    assert torch.equal(p1.data, p2.data)


def test_direction_indices_are_stable_and_device_local():
    p = nn.Parameter(torch.zeros(8))
    opt = _opt([p])
    device = torch.device("cpu")
    first = opt._direction_indices(5000, device, 1, 2)
    second = opt._direction_indices(5000, device, 1, 2)
    assert first.shape == (Adafactor._DIRECTION_EMA_SAMPLE,)
    assert first.dtype == torch.int64
    assert first.device == device
    assert torch.equal(first, second)
    assert opt._direction_indices(12, device, 0, 0) is None


def test_repeated_runs_match_dir_consistency():
    def run():
        torch.manual_seed(3)
        p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        opt = _opt([p])
        grad = torch.randn(64, 64)
        grad = grad / grad.norm()
        values = []
        for _ in range(5):
            p.grad = grad.to(dtype=torch.bfloat16)
            opt.step()
            values.append(opt.state[p]["dir_consistency"].detach().clone())
        return values

    first = run()
    second = run()
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_sketch_tracks_full_cosine_on_constant_grad():
    torch.manual_seed(4)
    p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
    grad = torch.randn(64, 64)
    grad = grad / grad.norm()
    opt = _opt([p], lr=1e-3)
    real_index = torch.Tensor.index_select
    captured = {}

    def wrapped(self, dim, index):
        if dim == 0 and index.numel() == Adafactor._DIRECTION_EMA_SAMPLE:
            captured["flat"] = self.detach().clone()
        return real_index(self, dim, index)

    torch.Tensor.index_select = wrapped
    ref_ema = None
    try:
        gap = None
        for _ in range(40):
            p.grad = grad.to(dtype=torch.bfloat16)
            opt.step()
            flat = captured["flat"]
            if ref_ema is None:
                ref_ema = torch.zeros_like(flat)
            ref_dc = torch.nn.functional.cosine_similarity(
                flat, ref_ema, dim=0, eps=1e-8
            )
            beta2t = opt.state[p]["beta2t"]
            ref_ema.mul_(beta2t).add_(flat, alpha=(1.0 - beta2t))
            sketch_dc = opt.state[p]["dir_consistency"].detach()
            gap = abs(float(sketch_dc) - float(ref_dc))
        buf = opt.state[p]["beta2_direction_ema"]
        assert buf.dtype == torch.float32
        assert tuple(buf.shape) == (Adafactor._DIRECTION_EMA_SAMPLE,)
        assert float(opt.state[p]["dir_consistency"]) > 0.9
        assert gap < 0.15
    finally:
        torch.Tensor.index_select = real_index


def test_load_drops_full_size_direction_ema():
    torch.manual_seed(5)
    p = nn.Parameter(torch.randn(2000))
    opt = _opt([p])
    p.grad = torch.randn_like(p)
    opt.step()
    state = opt.state_dict()
    for slot in state["state"].values():
        slot["beta2_direction_ema"] = torch.randn(2000)
    p2 = nn.Parameter(p.detach().clone())
    opt2 = _opt([p2])
    opt2.load_state_dict(state)
    assert "beta2_direction_ema" not in opt2.state[p2]
    p2.grad = torch.randn_like(p2)
    opt2.step()
    buf = opt2.state[p2]["beta2_direction_ema"]
    assert buf.dtype == torch.float32
    assert buf.numel() == Adafactor._DIRECTION_EMA_SAMPLE
    full = [
        key
        for key, value in opt2.state[p2].items()
        if isinstance(value, torch.Tensor) and value.numel() == p2.numel()
    ]
    assert full == ["exp_avg_sq"]


def test_load_restores_matching_fp32_sketch():
    torch.manual_seed(6)
    p = nn.Parameter(torch.randn(2000))
    opt = _opt([p])
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    saved = opt.state[p]["beta2_direction_ema"].detach().clone()
    state = opt.state_dict()
    p2 = nn.Parameter(p.detach().clone())
    opt2 = _opt([p2])
    opt2.load_state_dict(state)
    restored = opt2.state[p2]["beta2_direction_ema"]
    assert restored.dtype == torch.float32
    assert torch.equal(restored, saved)
    p2.grad = torch.randn_like(p2)
    opt2.step()
    assert opt2.state[p2]["beta2_direction_ema"].numel() == Adafactor._DIRECTION_EMA_SAMPLE


def _count_step_syncs(shape, device):
    item_n = 0
    bool_n = 0
    orig_item = torch.Tensor.item
    orig_bool = torch.Tensor.__bool__

    def _item(self):
        nonlocal item_n
        item_n += 1
        return orig_item(self)

    def _bool(self):
        nonlocal bool_n
        bool_n += 1
        return orig_bool(self)

    torch.Tensor.item = _item
    torch.Tensor.__bool__ = _bool
    try:
        p = nn.Parameter(torch.randn(*shape, device=device))
        opt = _opt([p], stochastic_rounding=False)
        p.grad = torch.randn_like(p)
        item_n = 0
        bool_n = 0
        opt.step()
        return item_n, bool_n
    finally:
        torch.Tensor.item = orig_item
        torch.Tensor.__bool__ = orig_bool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_cuda_sketch_adds_no_per_parameter_host_sync():
    small = _count_step_syncs((4, 3), "cuda")
    large = _count_step_syncs((64, 32), "cuda")
    assert large == small


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_sketch_indices_and_buffer_stay_on_device(monkeypatch):
    from toolkit.optimizers.optimizer_utils import copy_stochastic as orig_copy

    calls = {"n": 0}

    def _spy(target, source, eps=None):
        calls["n"] += 1
        assert target.dtype == torch.bfloat16
        orig_copy(target, source, eps)

    monkeypatch.setattr("toolkit.optimizers.adafactor.copy_stochastic", _spy)
    p = nn.Parameter(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
    opt = _opt([p], stochastic_rounding=True)
    indices = opt._direction_indices(p.numel(), p.device, 0, 0)
    assert indices.device == p.device
    grad = torch.randn(64, 64, device="cuda")
    for _ in range(3):
        p.grad = grad.to(dtype=torch.bfloat16)
        opt.step()
        buf = opt.state[p]["beta2_direction_ema"]
        assert buf.dtype == torch.float32
        assert buf.device == p.device
        assert buf.numel() == Adafactor._DIRECTION_EMA_SAMPLE
        dc = opt.state[p]["dir_consistency"]
        assert dc.device == p.device
    assert calls["n"] == 3
