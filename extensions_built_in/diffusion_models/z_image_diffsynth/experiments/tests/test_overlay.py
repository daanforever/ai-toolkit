"""overlay / parse_experiments / ema_horizon."""

from __future__ import annotations

import pytest

from ._loaders import load_overlay, recipe_path


@pytest.fixture(scope="module")
def overlay():
    return load_overlay()


@pytest.fixture(scope="module")
def recipe(overlay):
    return overlay.load_recipe(recipe_path())


def test_ema_horizon(overlay):
    assert overlay.ema_horizon(0.9) == 10
    assert overlay.ema_horizon(0.99) == 100


def test_parse_experiments(overlay, recipe):
    exp = overlay.parse_experiments(recipe["config"]["process"][0])
    assert exp["training_seed"] == 4
    assert exp["cases"][0]["id"] == "lr_vs_beta2"
    assert exp["cases"][0]["beta2_hi"] == 0.99


def test_no_tune_key(recipe):
    assert "tune" not in recipe["config"]["process"][0]
    assert "experiments" in recipe["config"]["process"][0]


def test_overlay_warm(overlay, recipe, tmp_path):
    exp = overlay.parse_experiments(recipe["config"]["process"][0])
    case = exp["cases"][0]
    tf = str(tmp_path / "warm").replace("\\", "/")
    cfg = overlay.overlay_run(
        recipe,
        case=case,
        lr=1e-4,
        beta2=0.99,
        steps=100,
        training_folder=tf,
        is_warm=True,
    )
    p0 = cfg["config"]["process"][0]
    assert cfg["config"]["name"] == "probe"
    assert p0["train"]["lr"] == 1e-4
    assert p0["train"]["steps"] == 100
    assert p0["train"]["optimizer_params"]["beta2"] == 0.99
    assert p0["train"]["optimizer_params"]["warmup_init"] is False
    assert p0["train"]["optimizer_params"]["weight_decay"] == 0.0
    assert p0["save"]["dtype"] == "fp32"
    assert p0["training_seed"] == 4
    assert p0["network"]["linear"] == 4
    assert p0["train"]["disable_sampling"] is True


def test_overlay_fork_steps(overlay, recipe, tmp_path):
    exp = overlay.parse_experiments(recipe["config"]["process"][0])
    case = exp["cases"][0]
    tf = str(tmp_path / "fork").replace("\\", "/")
    cfg = overlay.overlay_run(
        recipe,
        case=case,
        lr=4e-4,
        beta2=0.99,
        steps=110,
        training_folder=tf,
        is_warm=False,
    )
    p0 = cfg["config"]["process"][0]
    assert p0["train"]["steps"] == 110
    assert p0["train"]["lr"] == 4e-4


def test_strip_experiments(overlay, recipe, tmp_path):
    exp = overlay.parse_experiments(recipe["config"]["process"][0])
    case = exp["cases"][0]
    cfg = overlay.overlay_run(
        recipe,
        case=case,
        lr=1e-4,
        beta2=0.99,
        steps=100,
        training_folder=str(tmp_path),
        is_warm=True,
    )
    stripped = overlay.strip_experiments(cfg)
    assert "experiments" not in stripped["config"]["process"][0]
