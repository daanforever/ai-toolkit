"""Unit tests for runtime sample prompt apply / recache."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer


def _conn_returning(value):
    class _Cur:
        def execute(self, *a, **k):
            return None

        def fetchone(self):
            return value

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _Cur()

    return _Conn()


def _make_trainer(
    *,
    prompts_db,
    sample_prompts,
    caching=False,
    unload_te=False,
):
    """Build a DiffusionTrainer-like object without running __init__."""
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.is_ui_trainer = True
    t.job_id = "job-test"
    t.device_torch = "cpu"
    t._last_applied_runtime_prompts = None
    t.is_caching_text_embeddings = caching
    t.train_config = SimpleNamespace(unload_text_encoder=unload_te)
    samples = [SimpleNamespace(prompt=p) for p in sample_prompts]
    t.sample_config = SimpleNamespace(samples=samples)
    t.sd = SimpleNamespace(
        unet=MagicMock(),
        text_encoder_to=MagicMock(),
        _sampling_transformer=MagicMock(),
    )
    t.cache_sample_prompts = MagicMock()
    t._recache_sample_prompts_runtime = MagicMock()

    def _get():
        return prompts_db

    t.get_runtime_prompts = _get
    return t


def test_get_runtime_prompts_parses_json():
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.is_ui_trainer = True
    t.job_id = "j1"
    t._db_connect = lambda: _conn_returning((json.dumps(["a", "b"]),))
    assert t.get_runtime_prompts() == ["a", "b"]


def test_get_runtime_prompts_invalid_json():
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.is_ui_trainer = True
    t.job_id = "j1"
    t._db_connect = lambda: _conn_returning(("not-json",))
    assert t.get_runtime_prompts() is None


def test_apply_same_strings_skips_recache():
    t = _make_trainer(
        prompts_db=["hello", "world"],
        sample_prompts=["hello", "world"],
        caching=True,
    )
    t.apply_runtime_prompts()
    assert t._last_applied_runtime_prompts == ("hello", "world")
    t._recache_sample_prompts_runtime.assert_not_called()
    t.cache_sample_prompts.assert_not_called()


def test_apply_changed_strings_updates_by_index():
    t = _make_trainer(
        prompts_db=["new1", "new2", "extra-ignored"],
        sample_prompts=["old1", "old2"],
        caching=False,
        unload_te=False,
    )
    t.apply_runtime_prompts()
    assert t.sample_config.samples[0].prompt == "new1"
    assert t.sample_config.samples[1].prompt == "new2"
    assert len(t.sample_config.samples) == 2


def test_apply_shorter_list_leaves_remaining_unchanged():
    t = _make_trainer(
        prompts_db=["only-first"],
        sample_prompts=["a", "b"],
        caching=False,
    )
    t.apply_runtime_prompts()
    assert t.sample_config.samples[0].prompt == "only-first"
    assert t.sample_config.samples[1].prompt == "b"


def test_apply_with_cache_calls_recache():
    t = _make_trainer(
        prompts_db=["x"],
        sample_prompts=["y"],
        caching=True,
    )
    t.apply_runtime_prompts()
    assert t.sample_config.samples[0].prompt == "x"
    t._recache_sample_prompts_runtime.assert_called_once()


def test_apply_without_cache_no_te_reload():
    t = _make_trainer(
        prompts_db=["x"],
        sample_prompts=["y"],
        caching=False,
        unload_te=False,
    )
    t.apply_runtime_prompts()
    t._recache_sample_prompts_runtime.assert_not_called()
    t.sd.text_encoder_to.assert_not_called()


def test_recache_sequence_cache_path(monkeypatch):
    t = _make_trainer(prompts_db=["x"], sample_prompts=["y"], caching=True)
    t._recache_sample_prompts_runtime = (
        DiffusionTrainer._recache_sample_prompts_runtime.__get__(t, DiffusionTrainer)
    )
    t._offload_transformers_for_te_recache = MagicMock()
    t._restore_training_transformer_after_te_recache = MagicMock()
    t.cache_sample_prompts = MagicMock()
    t.sd.text_encoder_to = MagicMock()

    import toolkit.unloader as unloader

    reload_mock = MagicMock()
    unload_mock = MagicMock()
    monkeypatch.setattr(unloader, "reload_text_encoder", reload_mock)
    monkeypatch.setattr(unloader, "unload_text_encoder", unload_mock)

    t._recache_sample_prompts_runtime()

    t._offload_transformers_for_te_recache.assert_called_once()
    reload_mock.assert_called_once_with(t.sd)
    t.sd.text_encoder_to.assert_any_call(t.device_torch)
    t.cache_sample_prompts.assert_called_once()
    unload_mock.assert_called_once_with(t.sd)
    t._restore_training_transformer_after_te_recache.assert_called_once()


def test_recache_unload_only_path(monkeypatch):
    t = _make_trainer(
        prompts_db=["x"], sample_prompts=["y"], caching=False, unload_te=True
    )
    t._recache_sample_prompts_runtime = (
        DiffusionTrainer._recache_sample_prompts_runtime.__get__(t, DiffusionTrainer)
    )
    t._offload_transformers_for_te_recache = MagicMock()
    t._restore_training_transformer_after_te_recache = MagicMock()
    t.cache_sample_prompts = MagicMock()
    t.sd.text_encoder_to = MagicMock()

    import toolkit.unloader as unloader

    reload_mock = MagicMock()
    unload_mock = MagicMock()
    monkeypatch.setattr(unloader, "reload_text_encoder", reload_mock)
    monkeypatch.setattr(unloader, "unload_text_encoder", unload_mock)

    t._recache_sample_prompts_runtime()

    reload_mock.assert_not_called()
    unload_mock.assert_not_called()
    assert t.sd.text_encoder_to.call_args_list[0].args[0] == t.device_torch
    assert t.sd.text_encoder_to.call_args_list[-1].args[0] == "cpu"
    t.cache_sample_prompts.assert_called_once()


def test_reset_last_applied_clears_prompts():
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t._last_applied_runtime_lr = 1.0
    t._last_applied_runtime_gaussian_mean = None
    t._last_applied_runtime_gaussian_std = None
    t._last_applied_runtime_gaussian_mean_2 = None
    t._last_applied_runtime_gaussian_std_2 = None
    t._last_applied_runtime_weight_decay = None
    t._last_applied_runtime_weight_decay_increment = None
    t._last_applied_runtime_weight_decay_mode = None
    t._last_applied_runtime_beta1 = None
    t._last_applied_runtime_beta2 = None
    t._last_applied_runtime_content_or_style = None
    t._last_applied_runtime_timestep_type = None
    t._last_applied_runtime_timestep_weighting = None
    t._last_applied_runtime_network_weights = None
    t._last_applied_runtime_prompts = ("a",)
    t._last_applied_runtime_batch_size = None
    t._last_applied_runtime_gradient_accumulation = None
    t._last_applied_runtime_save_every = None
    t._last_applied_runtime_sample_every = None
    t._last_applied_runtime_warmup_steps = None
    t._last_applied_runtime_warmup_boost = None
    t._last_applied_runtime_min_snr_gamma = None
    t._last_applied_runtime_debug = None
    t._last_applied_runtime_fc_key = None
    t._last_applied_runtime_turbo_prior_steps = None
    t._last_applied_runtime_turbo_t_jitter = None
    DiffusionTrainer._reset_last_applied_runtime(t)
    assert t._last_applied_runtime_prompts is None
