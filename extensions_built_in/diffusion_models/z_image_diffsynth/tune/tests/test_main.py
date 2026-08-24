"""main() all-fail, safe_range funnel, and promote_top_k."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ._loaders import load_driver, load_probe

_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0c"
    b"IDATx\x9cchhh\x00\x00\x03\x04\x01\x81K\xd3\xd2\x10"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture(scope="module")
def driver():
    return load_driver()


@pytest.fixture(scope="module")
def probe_mod():
    return load_probe()


def _seed_test_train(root: Path) -> None:
    cache = root / "temp" / "test_train"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "ref.png").write_bytes(_MIN_PNG)
    (cache / "ref.txt").write_text("dog\n", encoding="utf-8")


def _seed_dataset(root: Path) -> None:
    ds = root / "datasets"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "img.png").write_bytes(_MIN_PNG)


def _ok_health():
    return SimpleNamespace(
        ok=True,
        reason=None,
        last_instability=0.0,
        loss_tag="loss",
        loss_first_mean=0.1,
        loss_last_mean=0.1,
        last_lr=1e-4,
        last_effective_lr=1e-4,
    )


def _visual(*, drop: bool, reason: str | None, score: float = 0.5):
    return SimpleNamespace(
        drop=drop,
        reason=reason,
        score=score,
        clip_i_s_r=0.0,
        clip_i_s_m=0.0,
        clip_i_m_r=0.0,
        clip_t=0.0,
        lpips_s_r=0.0,
        lpips_s_m=0.0,
        lpips_m_r=0.0,
    )


def _fake_run_probe_ok(probe_mod, seen_lrs: list | None = None):
    def fake_run_probe(config, *, python_exe=None, **kwargs):
        p0 = config["config"]["process"][0]
        tf = Path(p0["training_folder"])
        steps = int(p0["train"]["steps"])
        lr = float(p0["train"]["lr"])
        if seen_lrs is not None:
            seen_lrs.append(lr)
        save_root = tf / "probe"
        samples = save_root / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        if p0["train"].get("force_first_sample"):
            (samples / "0__000000000_0.png").write_bytes(_MIN_PNG)
        (samples / f"0__{steps:09d}_0.png").write_bytes(_MIN_PNG)
        return probe_mod.ProbeResult(
            exit_code=0,
            training_folder=str(tf),
            log_dir=p0["log_dir"],
            save_root=str(save_root),
            stdout="",
            stderr="",
        )

    return fake_run_probe


def test_main_all_fail_no_recommended(driver, probe_mod, monkeypatch, tmp_path, capsys):
    _seed_test_train(tmp_path)
    monkeypatch.setattr(driver, "_toolkit_root", lambda: tmp_path)

    def fake_run_probe(config, *, python_exe=None, **kwargs):
        p0 = config["config"]["process"][0]
        tf = p0["training_folder"]
        return probe_mod.ProbeResult(
            exit_code=1,
            training_folder=tf,
            log_dir=p0["log_dir"],
            save_root=str(Path(tf) / "probe"),
            stdout="",
            stderr="fail",
        )

    monkeypatch.setattr(driver, "run_probe", fake_run_probe)

    code = driver.main()
    captured = capsys.readouterr()

    assert code == 1
    assert "train.lr:" not in captured.out
    rec = tmp_path / "temp" / "tune"
    run_dirs = list(rec.iterdir()) if rec.is_dir() else []
    assert run_dirs, "expected scratch under tmp_path/temp/tune"
    for run_dir in run_dirs:
        recommended = run_dir / "recommended.json"
        if recommended.is_file():
            data = json.loads(recommended.read_text(encoding="utf-8"))
            assert "train.lr" not in data
            assert not data


def test_main_dead_before_safe_range_survives(
    driver, probe_mod, monkeypatch, tmp_path, capsys
):
    """dead at ckpt 10 then ok at 100 → LR survives; stage c writes recommended."""
    _seed_dataset(tmp_path)
    monkeypatch.setattr(driver, "_toolkit_root", lambda: tmp_path)
    monkeypatch.setattr(
        driver, "_copy_master", lambda step0, dest: shutil.copy2(step0, dest)
    )

    real_parse = driver.parse_tune

    def fake_parse(process0):
        tune = real_parse(process0)
        tune["stages"] = ["c"]
        tune["lrs"] = [1.0e-3]
        tune["c"] = {
            **tune["c"],
            "checkpoints": [10, 100],
        }
        return tune

    monkeypatch.setattr(driver, "parse_tune", fake_parse)
    monkeypatch.setattr(driver, "run_probe", _fake_run_probe_ok(probe_mod))
    monkeypatch.setattr(driver, "health_from_tb", lambda *a, **k: _ok_health())

    def fake_visual(*, sample, reference, master, caption, stage_id, dataset_images, prompt, thresholds):
        step = driver._parse_step(Path(sample).name)
        if step is not None and step < 100:
            return _visual(drop=True, reason="dead", score=0.1)
        return _visual(drop=False, reason=None, score=0.8)

    monkeypatch.setattr(driver, "visual_score", fake_visual)

    code = driver.main()
    captured = capsys.readouterr()

    assert code == 0
    assert "train.lr:" in captured.out
    rec = tmp_path / "temp" / "tune"
    run_dirs = list(rec.iterdir())
    assert run_dirs
    recommended = run_dirs[0] / "recommended.json"
    assert recommended.is_file()
    data = json.loads(recommended.read_text(encoding="utf-8"))
    assert data["train.lr"] == pytest.approx(1.0e-3)


def test_main_all_dead_at_safe_range_expands_then_exits(
    driver, probe_mod, monkeypatch, tmp_path, capsys
):
    """All dead at >= safe_range → one √10 expand (lr > 1e-3, cap 1e-2); second all-dead → exit 1."""
    _seed_dataset(tmp_path)
    monkeypatch.setattr(driver, "_toolkit_root", lambda: tmp_path)
    monkeypatch.setattr(
        driver, "_copy_master", lambda step0, dest: shutil.copy2(step0, dest)
    )

    real_parse = driver.parse_tune

    def fake_parse(process0):
        tune = real_parse(process0)
        tune["stages"] = ["c"]
        tune["lrs"] = [1.0e-3]
        tune["c"] = {
            **tune["c"],
            "checkpoints": [100],
        }
        return tune

    monkeypatch.setattr(driver, "parse_tune", fake_parse)

    seen_lrs: list[float] = []
    monkeypatch.setattr(driver, "run_probe", _fake_run_probe_ok(probe_mod, seen_lrs))
    monkeypatch.setattr(driver, "health_from_tb", lambda *a, **k: _ok_health())
    monkeypatch.setattr(
        driver,
        "visual_score",
        lambda **k: _visual(drop=True, reason="dead", score=0.0),
    )

    code = driver.main()
    captured = capsys.readouterr()

    assert code == 1
    assert "train.lr:" not in captured.out
    assert any(lr > 1.0e-3 for lr in seen_lrs)
    assert all(lr <= 1.0e-2 + 1e-12 for lr in seen_lrs)
    assert any(abs(lr - 1.0e-2) < 1e-9 or abs(lr - (1.0e-3 * (10**0.5))) < 1e-9 or lr > 1.0e-3 for lr in seen_lrs)


def test_promote_top_k_keeps_top_3(driver):
    survivors = [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]
    # Higher score ranks first; top 3 should be 1e-3, 3e-4, 1e-4
    last_by_lr = {
        1.0e-5: {"score": 0.10, "last_instability": 0.0},
        3.0e-5: {"score": 0.20, "last_instability": 0.0},
        1.0e-4: {"score": 0.50, "last_instability": 0.0},
        3.0e-4: {"score": 0.70, "last_instability": 0.0},
        1.0e-3: {"score": 0.90, "last_instability": 0.0},
    }
    promoted = driver._promote(survivors, last_by_lr, 3)
    assert promoted == [1.0e-3, 3.0e-4, 1.0e-4]
