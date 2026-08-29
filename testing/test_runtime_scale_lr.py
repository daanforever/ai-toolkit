"""Tests for runtime Gaussian scale_lr mean/std/mask SQLite bridge."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from extensions_built_in.sd_trainer.DiffusionTrainer import (
    DiffusionTrainer,
    RuntimeScaleLrMaskRead,
)


def _conn_returning(row):
    class _Cur:
        def execute(self, *a, **k):
            return None

        def fetchone(self):
            return row

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _Cur()

    return _Conn()


def _conn_raising_operational_error():
    class _Cur:
        def execute(self, *a, **k):
            raise sqlite3.OperationalError("no such column")

        def fetchone(self):
            return None

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _Cur()

    return _Conn()


def _make_trainer(*, optimizer=None):
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.is_ui_trainer = True
    t.job_id = "job-scale-lr"
    t.optimizer = optimizer if optimizer is not None else MagicMock()
    t._last_applied_runtime_scale_lr_by_index = None
    t._last_applied_runtime_scale_lr_config = None
    return t


def test_get_runtime_scale_lr_mask_valid_json():
    t = _make_trainer()
    t._db_connect = lambda: _conn_returning((json.dumps(["layers", "attn"]),))
    assert t.get_runtime_scale_lr_mask() == ["layers", "attn"]
    assert t._read_runtime_scale_lr_mask() == RuntimeScaleLrMaskRead(
        "ok", ["layers", "attn"]
    )


def test_get_runtime_scale_lr_mask_empty_array():
    t = _make_trainer()
    t._db_connect = lambda: _conn_returning((json.dumps([]),))
    assert t.get_runtime_scale_lr_mask() == []
    assert t._read_runtime_scale_lr_mask() == RuntimeScaleLrMaskRead("ok", [])


def test_get_runtime_scale_lr_mask_malformed_json():
    t = _make_trainer()
    t._db_connect = lambda: _conn_returning(("{not-json",))
    assert t._read_runtime_scale_lr_mask() == RuntimeScaleLrMaskRead("invalid")
    assert t.get_runtime_scale_lr_mask() is None


def test_get_runtime_scale_lr_mask_non_string_elements():
    t = _make_trainer()
    t._db_connect = lambda: _conn_returning((json.dumps(["layers", 1]),))
    assert t._read_runtime_scale_lr_mask() == RuntimeScaleLrMaskRead("invalid")
    assert t.get_runtime_scale_lr_mask() is None


def test_get_runtime_scale_lr_mask_absent_null():
    t = _make_trainer()
    t._db_connect = lambda: _conn_returning((None,))
    assert t._read_runtime_scale_lr_mask() == RuntimeScaleLrMaskRead("absent")
    assert t.get_runtime_scale_lr_mask() is None


def test_get_runtime_scale_lr_missing_columns():
    t = _make_trainer()
    t._db_connect = lambda: _conn_raising_operational_error()
    assert t.get_runtime_scale_lr_mean() is None
    assert t.get_runtime_scale_lr_std() is None
    assert t.get_runtime_scale_lr_mask() is None
    assert t._read_runtime_scale_lr_mask() == RuntimeScaleLrMaskRead("absent")
    assert t.get_runtime_scale_lr_by_index() is None


def test_get_runtime_scale_lr_mean_std_malformed_non_numeric():
    t = _make_trainer()
    t._db_connect = lambda: _conn_returning(("not-a-number",))
    assert t.get_runtime_scale_lr_mean() is None
    assert t.get_runtime_scale_lr_std() is None


def test_apply_malformed_mean_does_not_change_optimizer():
    opt = MagicMock()
    opt.optimizer = None
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t._db_connect = lambda: _conn_returning(("bad",))
    # mean/std getters hit the same mocked row; override std/mask/by_index separately
    t.get_runtime_scale_lr_std = lambda: 0.25
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("ok", ["layers"])
    t.get_runtime_scale_lr_by_index = lambda: True

    assert t.get_runtime_scale_lr_mean() is None
    t.apply_runtime_scale_lr()

    opt.set_scale_lr_config.assert_not_called()
    opt.set_scale_lr_by_index.assert_not_called()


def test_apply_malformed_std_does_not_change_optimizer():
    opt = MagicMock()
    opt.optimizer = None
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t.get_runtime_scale_lr_mean = lambda: 1.0
    t._db_connect = lambda: _conn_returning(("nope",))
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("ok", [])
    t.get_runtime_scale_lr_by_index = lambda: True

    assert t.get_runtime_scale_lr_std() is None
    t.apply_runtime_scale_lr()

    opt.set_scale_lr_config.assert_not_called()
    opt.set_scale_lr_by_index.assert_not_called()


def test_apply_valid_config_then_by_index_order():
    opt = MagicMock()
    opt.optimizer = None
    opt.scale_lr_by_index = False
    calls = []

    def set_config(mean, std, mask):
        calls.append(("config", mean, std, list(mask)))

    def set_by_index(value):
        calls.append(("by_index", value))

    opt.set_scale_lr_config = set_config
    opt.set_scale_lr_by_index = set_by_index

    t = _make_trainer(optimizer=opt)
    t.get_runtime_scale_lr_mean = lambda: 1.5
    t.get_runtime_scale_lr_std = lambda: 0.25
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("ok", ["layers"])
    t.get_runtime_scale_lr_by_index = lambda: True

    t.apply_runtime_scale_lr()

    assert calls == [("config", 1.5, 0.25, ["layers"]), ("by_index", True)]
    assert t._last_applied_runtime_scale_lr_config == (1.5, 0.25, ("layers",))
    assert t._last_applied_runtime_scale_lr_by_index is True


def test_apply_empty_mask_cache_key():
    opt = MagicMock()
    opt.optimizer = None
    opt.scale_lr_by_index = False
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t.get_runtime_scale_lr_mean = lambda: 0.0
    t.get_runtime_scale_lr_std = lambda: 0.5
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("ok", [])
    t.get_runtime_scale_lr_by_index = lambda: False

    t.apply_runtime_scale_lr()
    assert t._last_applied_runtime_scale_lr_config == (0.0, 0.5, ())
    opt.set_scale_lr_config.assert_called_once_with(0.0, 0.5, [])


def test_apply_repeated_cache_skips_optimizer():
    opt = MagicMock()
    opt.optimizer = None
    opt.scale_lr_by_index = True
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t.get_runtime_scale_lr_mean = lambda: 2.0
    t.get_runtime_scale_lr_std = lambda: 0.1
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("ok", ["attn"])
    t.get_runtime_scale_lr_by_index = lambda: True
    t._last_applied_runtime_scale_lr_config = (2.0, 0.1, ("attn",))
    t._last_applied_runtime_scale_lr_by_index = True

    t.apply_runtime_scale_lr()

    opt.set_scale_lr_config.assert_not_called()
    opt.set_scale_lr_by_index.assert_not_called()


def test_apply_malformed_mask_does_not_change_optimizer():
    opt = MagicMock()
    opt.optimizer = None
    opt.scale_lr_by_index = False
    opt.scale_lr_mean = None
    opt.scale_lr_std = None
    opt.scale_lr_mask = None
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t.get_runtime_scale_lr_mean = lambda: 1.0
    t.get_runtime_scale_lr_std = lambda: 0.2
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("invalid")
    t.get_runtime_scale_lr_by_index = lambda: True

    t.apply_runtime_scale_lr()

    opt.set_scale_lr_config.assert_not_called()
    opt.set_scale_lr_by_index.assert_not_called()
    assert t._last_applied_runtime_scale_lr_config is None
    assert t._last_applied_runtime_scale_lr_by_index is None


def test_apply_non_string_mask_elements_no_partial_application():
    opt = MagicMock()
    opt.optimizer = None
    opt.scale_lr_by_index = False
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t.get_runtime_scale_lr_mean = lambda: 1.0
    t.get_runtime_scale_lr_std = lambda: 0.2
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("invalid")
    t.get_runtime_scale_lr_by_index = lambda: True

    t.apply_runtime_scale_lr()

    opt.set_scale_lr_config.assert_not_called()
    opt.set_scale_lr_by_index.assert_not_called()


def test_leftover_by_index_true_without_mean_std_no_change():
    opt = MagicMock()
    opt.optimizer = None
    opt.scale_lr_by_index = False
    opt.scale_lr_mean = None
    opt.scale_lr_std = None
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t.get_runtime_scale_lr_mean = lambda: None
    t.get_runtime_scale_lr_std = lambda: None
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("absent")
    t.get_runtime_scale_lr_by_index = lambda: True

    t.apply_runtime_scale_lr()

    opt.set_scale_lr_config.assert_not_called()
    opt.set_scale_lr_by_index.assert_not_called()
    assert opt.scale_lr_by_index is False
    assert t._last_applied_runtime_scale_lr_by_index is None


def test_incomplete_mean_std_no_partial_application():
    opt = MagicMock()
    opt.optimizer = None
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t.get_runtime_scale_lr_mean = lambda: 1.0
    t.get_runtime_scale_lr_std = lambda: None
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("ok", ["layers"])
    t.get_runtime_scale_lr_by_index = lambda: True

    t.apply_runtime_scale_lr()

    opt.set_scale_lr_config.assert_not_called()
    opt.set_scale_lr_by_index.assert_not_called()


def test_missing_columns_no_runtime_override():
    opt = MagicMock()
    opt.optimizer = None
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t._db_connect = lambda: _conn_raising_operational_error()

    t.apply_runtime_scale_lr()

    opt.set_scale_lr_config.assert_not_called()
    opt.set_scale_lr_by_index.assert_not_called()


def test_reset_last_applied_clears_scale_lr_cache():
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t._last_applied_runtime_lr = None
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
    t._last_applied_runtime_prompts = None
    t._last_applied_runtime_batch_size = None
    t._last_applied_runtime_gradient_accumulation = None
    t._last_applied_runtime_save_every = None
    t._last_applied_runtime_sample_every = None
    t._last_applied_runtime_warmup_steps = None
    t._last_applied_runtime_warmup_boost = None
    t._last_applied_runtime_scale_lr_by_index = True
    t._last_applied_runtime_scale_lr_config = (1.0, 0.2, ("layers",))
    t._last_applied_runtime_min_snr_gamma = None
    t._last_applied_runtime_debug = None
    t._last_applied_runtime_fc_key = None
    t._last_applied_runtime_turbo_prior_steps = None
    t._last_applied_runtime_turbo_t_jitter = None
    t._last_applied_runtime_turbo_teacher_weight = None

    DiffusionTrainer._reset_last_applied_runtime(t)

    assert t._last_applied_runtime_scale_lr_by_index is None
    assert t._last_applied_runtime_scale_lr_config is None


def test_null_mask_defaults_to_empty_tuple_cache():
    opt = MagicMock()
    opt.optimizer = None
    opt.scale_lr_by_index = False
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t.get_runtime_scale_lr_mean = lambda: 3.0
    t.get_runtime_scale_lr_std = lambda: 0.4
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("absent")
    t.get_runtime_scale_lr_by_index = lambda: False

    t.apply_runtime_scale_lr()

    opt.set_scale_lr_config.assert_called_once_with(3.0, 0.4, [])
    assert t._last_applied_runtime_scale_lr_config == (3.0, 0.4, ())
