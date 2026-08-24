"""visual_score gates (monkeypatched metrics; no CLIP/torch download)."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from ._loaders import load_rubric


@pytest.fixture(scope="module")
def rubric():
    return load_rubric()


@pytest.fixture
def block_torch_and_clip(monkeypatch):
    real_import = builtins.__import__

    def _guard(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root in ("torch", "open_clip", "lpips"):
            raise AssertionError(f"forbidden import during visual test: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guard)


def _patch_metrics(rubric, monkeypatch, *, clip_i, clip_t, lpips, clip_i_ds=None):
    monkeypatch.setattr(rubric, "_metric_clip_i", clip_i)
    monkeypatch.setattr(rubric, "_metric_clip_t", clip_t)
    monkeypatch.setattr(rubric, "_metric_lpips", lpips)
    if clip_i_ds is not None:
        monkeypatch.setattr(rubric, "_metric_clip_i_vs_dataset", clip_i_ds)


def test_visual_dead_gate_stage_a(rubric, monkeypatch, block_torch_and_clip, tmp_path):
    sample = tmp_path / "s.png"
    reference = tmp_path / "r.png"
    master = tmp_path / "m.png"
    for p in (sample, reference, master):
        p.write_bytes(b"x")

    # LPIPS(S,M) < dead (0.04) and CLIP-I(S,R) <= CLIP-I(M,R) + 0.01
    def fake_lpips(a, b):
        if Path(a) == sample and Path(b) == master:
            return 0.01
        if Path(a) == sample and Path(b) == reference:
            return 0.2
        if Path(a) == master and Path(b) == reference:
            return 0.3
        return 0.5

    def fake_clip_i(a, b):
        if Path(a) == sample and Path(b) == reference:
            return 0.50
        if Path(a) == master and Path(b) == reference:
            return 0.50
        if Path(a) == sample and Path(b) == master:
            return 0.99
        return 0.0

    _patch_metrics(
        rubric,
        monkeypatch,
        clip_i=fake_clip_i,
        clip_t=lambda *_a, **_k: 0.4,
        lpips=fake_lpips,
    )

    result = rubric.visual_score(
        sample=sample,
        reference=reference,
        master=master,
        caption="dog",
        stage_id="a",
        dataset_images=[],
        prompt="dog",
        thresholds={"lpips_dead": 0.04, "lpips_boom": 0.45},
    )
    assert result.drop is True
    assert result.reason == "dead"


def test_visual_exploded_gate_stage_a(rubric, monkeypatch, block_torch_and_clip, tmp_path):
    sample = tmp_path / "s.png"
    reference = tmp_path / "r.png"
    master = tmp_path / "m.png"
    for p in (sample, reference, master):
        p.write_bytes(b"x")

    # LPIPS(S,M) > boom and CLIP-I(S,R) < CLIP-I(M,R)
    def fake_lpips(a, b):
        if Path(a) == sample and Path(b) == master:
            return 0.60
        return 0.2

    def fake_clip_i(a, b):
        if Path(a) == sample and Path(b) == reference:
            return 0.40
        if Path(a) == master and Path(b) == reference:
            return 0.55
        return 0.1

    _patch_metrics(
        rubric,
        monkeypatch,
        clip_i=fake_clip_i,
        clip_t=lambda *_a, **_k: 0.3,
        lpips=fake_lpips,
    )

    result = rubric.visual_score(
        sample=sample,
        reference=reference,
        master=master,
        caption="dog",
        stage_id="a",
        dataset_images=[],
        prompt="dog",
        thresholds={"lpips_dead": 0.04, "lpips_boom": 0.45},
    )
    assert result.drop is True
    assert result.reason == "exploded"


def test_visual_survive_stage_a(rubric, monkeypatch, block_torch_and_clip, tmp_path):
    sample = tmp_path / "s.png"
    reference = tmp_path / "r.png"
    master = tmp_path / "m.png"
    for p in (sample, reference, master):
        p.write_bytes(b"x")

    def fake_lpips(a, b):
        if Path(a) == sample and Path(b) == master:
            return 0.20  # between dead and boom
        if Path(a) == sample and Path(b) == reference:
            return 0.15
        if Path(a) == master and Path(b) == reference:
            return 0.35
        return 0.2

    def fake_clip_i(a, b):
        if Path(a) == sample and Path(b) == reference:
            return 0.70
        if Path(a) == master and Path(b) == reference:
            return 0.50
        return 0.4

    _patch_metrics(
        rubric,
        monkeypatch,
        clip_i=fake_clip_i,
        clip_t=lambda *_a, **_k: 0.6,
        lpips=fake_lpips,
    )

    result = rubric.visual_score(
        sample=sample,
        reference=reference,
        master=master,
        caption="dog",
        stage_id="a",
        dataset_images=[],
        prompt="dog",
        thresholds={"lpips_dead": 0.04, "lpips_boom": 0.45},
    )
    assert result.drop is False
    assert result.reason is None
    # 0.45*0.70 + 0.20*0.6 + 0.25*max(0,0.20) + 0.10*max(0,0.20)
    expected = 0.45 * 0.70 + 0.20 * 0.6 + 0.25 * 0.20 + 0.10 * 0.20
    assert result.score == pytest.approx(expected)
