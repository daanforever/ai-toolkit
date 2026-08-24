"""overlay_probe / strip_tune / write_overlay_yaml."""

from __future__ import annotations

import pytest
import yaml

from ._loaders import load_overlay, recipe_path

@pytest.fixture(scope="module")
def overlay():
    return load_overlay()


@pytest.fixture(scope="module")
def recipe(overlay):
    return overlay.load_recipe(recipe_path())


def _p0(cfg: dict) -> dict:
    return cfg["config"]["process"][0]


def test_overlay_probe_stage_a_keys(overlay, recipe, tmp_path):
    training_folder = tmp_path / "trial_a"
    training_folder.mkdir()
    tf = str(training_folder).replace("\\", "/")

    cfg = overlay.overlay_probe(
        recipe,
        lr=3.0e-4,
        steps=10,
        stage_id="a",
        training_folder=tf,
        is_first_segment=True,
    )
    p0 = _p0(cfg)

    assert cfg["config"]["name"] == "probe"
    assert p0["train"]["lr"] == 3.0e-4
    assert p0["train"]["steps"] == 10
    assert p0["train"]["optimizer_params"]["warmup_steps"] == 8
    assert p0["train"]["skip_first_sample"] is False
    assert p0["train"]["force_first_sample"] is True
    assert p0["save"]["save_every"] == 10
    assert p0["sample"]["sample_every"] == 10
    assert p0["sample"]["width"] == 256
    assert p0["sample"]["height"] == 256
    assert p0["network"]["linear"] == 4
    assert p0["network"]["linear_alpha"] == 4
    assert p0["datasets"][0]["resolution"] == [512]
    assert p0["datasets"][0]["folder_path"].replace("\\", "/") == f"{tf}/ref"
    assert p0["log_dir"].replace("\\", "/") == f"{tf}/tb"
    assert p0["sqlite_db_path"].replace("\\", "/") == f"{tf}/aitk.db"
    assert p0["training_folder"].replace("\\", "/") == tf
    assert p0["train"]["timestep_type"] == "turbo_prior"
    assert p0["sample"]["guidance_scale"] == 0
    assert p0["sample"]["sample_steps"] == 8
    assert p0["train"]["content_or_style"] == "balanced"
    assert p0["train"]["turbo_teacher_weight"] is True
    assert p0["model"]["model_kwargs"]["use_diffsynth_training_loop"] is False
    assert p0["logging"]["use_ui_logger"] is False
    assert p0["logging"]["log_every"] != 1
    assert p0["performance_log_every"] == 0
    assert p0["train"]["gradient_checkpointing"] is True
    assert p0["model"]["compile"] is False
    assert p0["train"]["dtype"] == "bf16"
    assert p0["network"]["dtype"] == "fp32"
    assert p0["train"].get("disable_sampling") is not True
    assert p0["sample"]["sample_every"] == 10


def test_overlay_probe_forces_turbo_teacher_weight(overlay, recipe, tmp_path):
    """Probe train becomes True from tune even if recipe train starts False."""
    import copy

    bad = copy.deepcopy(recipe)
    bad["config"]["process"][0]["train"]["turbo_teacher_weight"] = False
    bad["config"]["process"][0]["tune"]["turbo_teacher_weight"] = True
    cfg = overlay.overlay_probe(
        bad,
        lr=1.0e-4,
        steps=10,
        stage_id="a",
        training_folder=str(tmp_path / "t"),
        is_first_segment=True,
    )
    p0 = _p0(cfg)
    assert p0["train"]["turbo_teacher_weight"] is True
    assert p0["train"]["timestep_type"] == "turbo_prior"
    stripped = overlay.strip_tune(cfg)
    assert "tune" not in _p0(stripped)
    assert _p0(stripped)["train"]["turbo_teacher_weight"] is True


def test_overlay_probe_stage_c_leaves_recipe_sample_wh(overlay, recipe, tmp_path):
    tf = str(tmp_path / "trial_c")
    cfg = overlay.overlay_probe(
        recipe,
        lr=1.0e-4,
        steps=100,
        stage_id="c",
        training_folder=tf,
        is_first_segment=True,
    )
    sample = _p0(cfg)["sample"]
    assert sample["width"] == 1024
    assert sample["height"] == 768


def test_overlay_probe_resume_segment(overlay, recipe, tmp_path):
    training_folder = tmp_path / "trial_resume"
    training_folder.mkdir()
    tf = str(training_folder)

    first = overlay.overlay_probe(
        recipe,
        lr=1.0e-4,
        steps=10,
        stage_id="a",
        training_folder=tf,
        is_first_segment=True,
    )
    second = overlay.overlay_probe(
        recipe,
        lr=1.0e-4,
        steps=100,
        stage_id="a",
        training_folder=tf,
        is_first_segment=False,
    )

    assert first["config"]["name"] == "probe"
    assert second["config"]["name"] == "probe"
    assert _p0(first)["training_folder"] == _p0(second)["training_folder"] == tf
    assert _p0(second)["train"]["steps"] == 100
    assert _p0(second)["train"]["steps"] > _p0(first)["train"]["steps"]
    assert _p0(second)["train"]["skip_first_sample"] is True
    assert _p0(second)["train"]["force_first_sample"] is False
    assert _p0(second)["save"]["save_every"] == 100
    assert _p0(second)["sample"]["sample_every"] == 100


def test_strip_tune_drops_tune(overlay, recipe, tmp_path):
    cfg = overlay.overlay_probe(
        recipe,
        lr=1.0e-4,
        steps=10,
        stage_id="a",
        training_folder=str(tmp_path / "t"),
        is_first_segment=True,
    )
    assert "tune" in _p0(cfg)
    stripped = overlay.strip_tune(cfg)
    assert "tune" not in _p0(stripped)
    assert "tune" in _p0(cfg)  # original untouched


def test_write_overlay_yaml_to_tmp_path(overlay, recipe, tmp_path):
    cfg = overlay.overlay_probe(
        recipe,
        lr=1.0e-4,
        steps=10,
        stage_id="a",
        training_folder=str(tmp_path / "t"),
        is_first_segment=True,
    )
    dest = tmp_path / "overlays" / "probe.yaml"
    out = overlay.write_overlay_yaml(cfg, dest)
    assert out == dest
    assert dest.is_file()
    loaded = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert loaded["config"]["name"] == "probe"
    # training_folder is pytest tmp_path, not system tempfile
    assert str(tmp_path) in str(_p0(cfg)["training_folder"])
