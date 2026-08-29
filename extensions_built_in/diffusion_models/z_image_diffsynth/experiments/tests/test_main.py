"""Driver with mocked run_probe (no GPU / run_job)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from ._loaders import load_driver, load_overlay, recipe_path

_WARM_BASE = {
    "diffusion_model.layers.0.attention.to_q.lora_down.weight": torch.zeros(2, 4),
    "diffusion_model.layers.0.attention.to_q.lora_up.weight": torch.zeros(4, 2),
}


def _fake_probe(save_root: Path, *, scale: float, warm_sd: dict):
    """Write LoRA weights = warm + same-direction scaled delta."""
    save_root.mkdir(parents=True, exist_ok=True)
    sd = {}
    for k, v in warm_sd.items():
        if "down" in k:
            sd[k] = warm_sd[k] + scale * 0.1
        else:
            sd[k] = warm_sd[k] + scale * 0.05
    save_file(sd, str(save_root / "probe.safetensors"))
    (save_root / "optimizer.pt").write_bytes(b"fake")
    return sd


def _scale_for(lr: float, beta2: float) -> float:
    if abs(lr - 1e-4) < 1e-12 and abs(beta2 - 0.99) < 1e-12:
        return 1.0  # continue
    if abs(lr - 4e-4) < 1e-12 and abs(beta2 - 0.99) < 1e-12:
        return 2.0  # lr_x4
    if abs(lr - 1e-4) < 1e-12 and abs(beta2 - 0.9) < 1e-12:
        return 2.0  # beta2_0.9 (close to lr_x4)
    if abs(lr - 4e-4) < 1e-12 and abs(beta2 - 0.9) < 1e-12:
        return 3.0  # both
    return 1.5 + 0.5 * (beta2 - 0.85)


def _plant_test_train(tmp_path, monkeypatch, driver):
    test_train = tmp_path / "temp" / "test_train"
    test_train.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(test_train / "000.png")
    (test_train / "000.txt").write_text("dog\n", encoding="utf-8")
    monkeypatch.setattr(driver, "_toolkit_root", lambda: tmp_path)


def _case_ctx():
    overlay = load_overlay()
    driver = load_driver()
    recipe = overlay.load_recipe(recipe_path())
    exp = overlay.parse_experiments(recipe["config"]["process"][0])
    case = exp["cases"][0]
    return driver, recipe, exp, case


def _write_session_loras(config, *, skip_ids=()):
    if isinstance(config, Path):
        raise AssertionError("expected dict config in mock")
    exp = config["config"]["process"][0]["experiments"]
    warm_root = Path(exp["warm_training_folder"]) / "probe"
    warm_root.mkdir(parents=True, exist_ok=True)
    save_file(_WARM_BASE, str(warm_root / "probe.safetensors"))
    (warm_root / "optimizer.pt").write_bytes(b"fake")
    for fork in exp["forks"]:
        if fork["id"] in skip_ids:
            continue
        scale = _scale_for(float(fork["lr"]), float(fork["beta2"]))
        _fake_probe(
            Path(fork["training_folder"]) / "probe",
            scale=scale,
            warm_sd=_WARM_BASE,
        )
    tf = Path(config["config"]["process"][0]["training_folder"])
    name = config["config"]["name"]
    return SimpleNamespace(
        exit_code=0,
        training_folder=str(tf),
        log_dir=str(tf / "tb"),
        save_root=str(tf / name),
        stdout="",
        stderr="",
    )


def test_run_case_mock(tmp_path, monkeypatch):
    driver, recipe, exp, case = _case_ctx()
    _plant_test_train(tmp_path, monkeypatch, driver)
    call_i = {"n": 0}

    def fake_run_probe(config, **kwargs):
        call_i["n"] += 1
        ns = _write_session_loras(config)
        ns.stdout = (
            "experiment: gpu=NVIDIA GeForce RTX 5080 torch_cuda=13.2\n"
        )
        return ns

    report = driver.run_case(
        recipe,
        exp,
        case,
        tmp_path / "run",
        run_probe_fn=fake_run_probe,
        python_exe=Path("python"),
    )
    assert report.get("error") is None
    assert report["prefix_steps"] == 100
    assert report["measure_steps"] == 10
    assert "continue" in report["forks"]
    assert "baseline" not in report["forks"]
    assert "calibrate" in report
    cal = report["calibrate"]
    assert "equivalence" in cal
    assert "equiv_ratio" in cal["equivalence"]
    assert cal["equivalence"]["status"] in {"equivalent", "partial", "divergent"}
    assert "s_lr" in cal
    assert "s_b2" in cal
    assert call_i["n"] == 1
    assert report.get("gpu") == "NVIDIA GeForce RTX 5080"

    md = driver.render_markdown_report(
        run_id="1787938487441",
        reports=[report],
        wall_s=503.6,
    )
    assert "lr_vs_beta2" in md
    assert "temp/experiments/1787938487441" in md
    assert report["calibrate"]["equivalence"]["status"] in md
    assert "NVIDIA GeForce RTX 5080" in md
    assert "~8.4 min" in md
    assert "equiv_ratio" in md

    dest = tmp_path / "package_report.md"
    monkeypatch.setattr(driver, "_package_report_md_path", lambda: dest)
    written = driver.write_package_report_md(
        run_id="1787938487441",
        reports=[report],
        wall_s=503.6,
    )
    assert written == dest
    assert dest.is_file()
    assert "NVIDIA GeForce RTX 5080" in dest.read_text(encoding="utf-8")

    case_dir = tmp_path / "run" / str(case["id"])
    assert (case_dir / "warm" / "probe" / "probe.safetensors").is_file()
    fork_dirs = sorted(case_dir.glob("fork_*"))
    assert fork_dirs
    for d in fork_dirs:
        assert (d / "probe" / "probe.safetensors").is_file()


def test_gpu_from_log():
    driver = load_driver()
    log = "experiment: gpu=NVIDIA GeForce RTX 5080 torch_cuda=13.2\n"
    assert driver._gpu_from_log(log) == "NVIDIA GeForce RTX 5080"
    assert driver._gpu_from_log("") is None


def test_train_missing_or_empty(tmp_path, monkeypatch):
    driver, recipe, exp, case = _case_ctx()
    assert case.get("dataset") == "one_ref"
    monkeypatch.setattr(driver, "_toolkit_root", lambda: tmp_path)
    report = driver.run_case(
        recipe,
        exp,
        case,
        tmp_path / "run",
        python_exe=Path("python"),
    )
    assert report.get("error") == "test_train_missing_or_empty"


def test_missing_continue_or_lr_x4(tmp_path, monkeypatch):
    driver, recipe, exp, case = _case_ctx()
    _plant_test_train(tmp_path, monkeypatch, driver)
    call_i = {"n": 0}

    def fake_run_probe(config, **kwargs):
        call_i["n"] += 1
        return _write_session_loras(config, skip_ids=("continue", "lr_x4"))

    report = driver.run_case(
        recipe,
        exp,
        case,
        tmp_path / "run",
        run_probe_fn=fake_run_probe,
        python_exe=Path("python"),
    )
    assert report.get("error") == "missing_continue_or_lr_x4"
    assert call_i["n"] == 1
    case_dir = tmp_path / "run" / str(case["id"])
    assert (case_dir / "warm" / "probe" / "probe.safetensors").is_file()
