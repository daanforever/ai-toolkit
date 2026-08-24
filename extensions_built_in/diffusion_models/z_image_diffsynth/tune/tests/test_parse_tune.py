"""parse_tune validation."""

from __future__ import annotations

import copy

import pytest

from ._loaders import load_overlay, recipe_path


@pytest.fixture(scope="module")
def overlay():
    return load_overlay()


@pytest.fixture(scope="module")
def default_process0(overlay):
    recipe = overlay.load_recipe(recipe_path())
    return recipe["config"]["process"][0]


def test_parse_tune_default_recipe_ok(overlay, default_process0):
    """Default recipe (b warmup 30 / ckpt 10; c warmup 100 / ckpt 100) must parse."""
    tune = overlay.parse_tune(default_process0)
    assert tune["b"]["warmup_steps"] == 30
    assert tune["b"]["checkpoints"][0] == 10
    assert tune["c"]["warmup_steps"] == 100
    assert tune["c"]["checkpoints"][0] == 100
    assert tune["promote_top_k"]["a"] == 3
    assert tune["promote_top_k"]["b"] == 2
    assert tune["safe_range"] == 100
    assert tune["step_timeout_s"] == 30.0
    proc = copy.deepcopy(default_process0)
    proc["tune"].pop("step_timeout_s", None)
    proc["tune"].pop("load_budget_s", None)
    proc["tune"].pop("sample_budget_s", None)
    tune = overlay.parse_tune(proc)
    assert tune["step_timeout_s"] == 2.0
    assert tune["load_budget_s"] == 180.0
    assert tune["sample_budget_s"] == 60.0


def test_parse_tune_safe_range_missing_defaults_100(overlay, default_process0):
    proc = copy.deepcopy(default_process0)
    proc["tune"].pop("safe_range", None)
    tune = overlay.parse_tune(proc)
    assert tune["safe_range"] == 100


def test_parse_tune_safe_range_explicit_kept(overlay, default_process0):
    proc = copy.deepcopy(default_process0)
    proc["tune"]["safe_range"] = 50
    tune = overlay.parse_tune(proc)
    assert tune["safe_range"] == 50


def test_parse_tune_checkpoints_strictly_increasing(overlay, default_process0):
    bad = copy.deepcopy(default_process0)
    bad["tune"]["a"]["checkpoints"] = [10, 10]
    with pytest.raises(ValueError, match="strictly increasing"):
        overlay.parse_tune(bad)

    bad2 = copy.deepcopy(default_process0)
    bad2["tune"]["a"]["checkpoints"] = [100, 10]
    with pytest.raises(ValueError, match="strictly increasing"):
        overlay.parse_tune(bad2)


def test_parse_tune_missing_promote_top_k_a_or_b_fails(overlay, default_process0):
    missing_a = copy.deepcopy(default_process0)
    missing_a["tune"]["promote_top_k"] = {"b": 2}
    with pytest.raises(ValueError, match="promote_top_k"):
        overlay.parse_tune(missing_a)

    missing_b = copy.deepcopy(default_process0)
    missing_b["tune"]["promote_top_k"] = {"a": 3}
    with pytest.raises(ValueError, match="promote_top_k"):
        overlay.parse_tune(missing_b)

    missing_all = copy.deepcopy(default_process0)
    del missing_all["tune"]["promote_top_k"]
    with pytest.raises(ValueError, match="promote_top_k"):
        overlay.parse_tune(missing_all)
