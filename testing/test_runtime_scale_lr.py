"""Tests for runtime Gaussian scale_lr mean/std/mask SQLite bridge."""

from __future__ import annotations

import json
import sqlite3
import sys

import pytest
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
    t._clear_sql = []

    def _db_connect():
        class _Cur:
            def execute(self, sql, params=None):
                t._clear_sql.append((sql, params))

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

    t._db_connect = _db_connect
    return t


def _cleared_columns(trainer):
    cols = []
    for sql, _params in trainer._clear_sql:
        if "UPDATE RuntimeParams SET" not in sql:
            continue
        assignment = sql.split("SET", 1)[1].split("WHERE", 1)[0]
        for part in assignment.split(","):
            name = part.strip().split("=", 1)[0].strip()
            if name:
                cols.append(name)
    return cols


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
    assert _cleared_columns(t) == [
        "runtime_scale_lr_mean",
        "runtime_scale_lr_std",
        "runtime_scale_lr_mask",
        "runtime_scale_lr_by_index",
    ]


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
    opt.set_scale_lr_config.assert_called_once_with(0.0, 0.5, [])
    assert "runtime_scale_lr_mask" in _cleared_columns(t)
    assert "DELETE" not in " ".join(sql for sql, _ in t._clear_sql)


def test_apply_repeated_cache_skips_optimizer():
    opt = MagicMock()
    opt.optimizer = None
    opt.scale_lr_by_index = True
    opt.scale_lr_mean = 2.0
    opt.scale_lr_std = 0.1
    opt.scale_lr_mask = ("attn",)
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t.get_runtime_scale_lr_mean = lambda: 2.0
    t.get_runtime_scale_lr_std = lambda: 0.1
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("ok", ["attn"])
    t.get_runtime_scale_lr_by_index = lambda: True

    t.apply_runtime_scale_lr()

    opt.set_scale_lr_config.assert_not_called()
    opt.set_scale_lr_by_index.assert_not_called()
    assert "runtime_scale_lr_mean" in _cleared_columns(t)


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
    assert _cleared_columns(t) == []


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
    assert _cleared_columns(t) == []


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


def test_apply_scale_lr_nulls_columns_and_keeps_row(tmp_path):
    db_path = tmp_path / "runtime.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE RuntimeParams ("
        "jobId TEXT PRIMARY KEY, runtime_scale_lr_mean REAL, "
        "runtime_scale_lr_std REAL, runtime_scale_lr_mask TEXT, "
        "runtime_scale_lr_by_index INTEGER)"
    )
    conn.execute(
        "INSERT INTO RuntimeParams VALUES (?, ?, ?, ?, ?)",
        ("job-scale-lr", 3.0, 0.4, None, 0),
    )
    conn.commit()
    conn.close()

    opt = MagicMock()
    opt.optimizer = None
    opt.scale_lr_by_index = True
    opt.scale_lr_mean = None
    opt.scale_lr_std = None
    opt.scale_lr_mask = None
    opt.set_scale_lr_config = MagicMock()
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t.sqlite_db_path = str(db_path)
    t._db_connect = DiffusionTrainer._db_connect.__get__(t, DiffusionTrainer)
    t.get_runtime_scale_lr_mean = lambda: 3.0
    t.get_runtime_scale_lr_std = lambda: 0.4
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("absent")
    t.get_runtime_scale_lr_by_index = lambda: False

    t.apply_runtime_scale_lr()

    opt.set_scale_lr_config.assert_called_once_with(3.0, 0.4, [])
    check = sqlite3.connect(db_path)
    row = check.execute(
        "SELECT runtime_scale_lr_mean, runtime_scale_lr_std, "
        "runtime_scale_lr_by_index, runtime_scale_lr_mask FROM RuntimeParams"
    ).fetchone()
    count = check.execute("SELECT COUNT(*) FROM RuntimeParams").fetchone()[0]
    check.close()
    assert row == (None, None, None, None)
    assert count == 1


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
    cleared = _cleared_columns(t)
    assert "runtime_scale_lr_mean" in cleared
    assert "runtime_scale_lr_std" in cleared
    assert "runtime_scale_lr_mask" not in cleared


def test_scale_config_value_error_does_not_clear():
    opt = MagicMock()
    opt.optimizer = None
    opt.scale_lr_by_index = False
    opt.scale_lr_mean = None
    opt.scale_lr_std = None
    opt.scale_lr_mask = None
    opt.set_scale_lr_config = MagicMock(side_effect=ValueError("bad mask"))
    opt.set_scale_lr_by_index = MagicMock()

    t = _make_trainer(optimizer=opt)
    t.get_runtime_scale_lr_mean = lambda: 1.0
    t.get_runtime_scale_lr_std = lambda: 0.2
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("ok", ["layers"])
    t.get_runtime_scale_lr_by_index = lambda: True

    t.apply_runtime_scale_lr()

    opt.set_scale_lr_by_index.assert_not_called()
    assert _cleared_columns(t) == []


def test_partial_gaussian_clears_only_consumed_column():
    from types import SimpleNamespace

    t = _make_trainer()
    t.train_config = SimpleNamespace(gaussian_mean=0.0, gaussian_std=0.5)
    t.get_runtime_gaussian_params = lambda: (1.25, None)
    t.apply_runtime_gaussian_params()
    assert t.train_config.gaussian_mean == 1.25
    assert t.train_config.gaussian_std == 0.5
    assert _cleared_columns(t) == ["runtime_gaussian_mean"]


def test_beta1_null_disables_only_after_numeric_consume():
    opt = MagicMock()
    opt.optimizer = None
    opt._beta1 = 0.9

    def set_beta1(value):
        opt._beta1 = value

    opt.set_beta1 = set_beta1
    t = _make_trainer(optimizer=opt)
    t.get_runtime_beta1 = lambda: None
    t.apply_runtime_beta1()
    assert opt._beta1 == 0.9
    assert _cleared_columns(t) == []

    t.get_runtime_beta1 = lambda: 0.5
    t.apply_runtime_beta1()
    assert opt._beta1 == 0.5
    assert _cleared_columns(t) == []

    t.get_runtime_beta1 = lambda: None
    t.apply_runtime_beta1()
    assert opt._beta1 is None
    assert _cleared_columns(t) == ["runtime_beta1"]

    t.apply_runtime_beta1()
    assert opt._beta1 is None


def test_train_start_hook_does_not_touch_runtime_params():
    t = _make_trainer()
    t.maybe_stop = lambda: None
    t.update_step = lambda: None
    t.update_status = lambda *a, **k: None
    t.timer = MagicMock()
    t.internal_hook_before_train_loop()
    assert t._clear_sql == []


def test_clear_runtime_columns_main_process_only():
    from types import SimpleNamespace

    t = _make_trainer()
    t.accelerator = SimpleNamespace(is_main_process=False)
    t._clear_runtime_columns("runtime_lr")
    assert t._clear_sql == []
    t.accelerator = SimpleNamespace(is_main_process=True)
    t._clear_runtime_columns("runtime_lr")
    assert _cleared_columns(t) == ["runtime_lr"]


def _step(opt, params):
    import torch

    for p in params:
        p.grad = torch.zeros_like(p)
    opt.step()


def test_completed_warmup_unrelated_apply_keeps_lr_and_scale_apply_ramps():
    import torch
    from types import SimpleNamespace
    from toolkit.optimizers.adafactor import Adafactor

    lr = 1e-4
    steps = 100
    p0 = torch.nn.Parameter(torch.ones(2))
    p1 = torch.nn.Parameter(torch.ones(2))
    params = (p0, p1)
    opt = Adafactor(
        [
            {"params": [p0], "index": 0, "name": "layers_0"},
            {"params": [p1], "index": 1, "name": "layers_1"},
        ],
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=steps,
        beta1=None,
        weight_decay=0.0,
        factored=False,
    )
    for _ in range(steps + 2):
        _step(opt, params)

    g1 = opt.param_groups[1]
    assert opt.get_learning_rates()[1] == pytest.approx(lr)
    assert g1.get("warmup_active") is False
    assert "warmup_start" not in g1
    prev = g1["warmup_lr_previous"]
    assert prev == pytest.approx(lr)
    warmup_keys = {
        k: g1[k] for k in g1 if k.startswith("warmup_")
    }

    calls = []
    real_config = opt.set_scale_lr_config
    real_by_index = opt.set_scale_lr_by_index

    def spy_config(*args, **kwargs):
        calls.append("config")
        return real_config(*args, **kwargs)

    def spy_by_index(*args, **kwargs):
        calls.append("by_index")
        return real_by_index(*args, **kwargs)

    opt.set_scale_lr_config = spy_config
    opt.set_scale_lr_by_index = spy_by_index

    t = _make_trainer(optimizer=opt)
    t.train_config = SimpleNamespace(
        batch_size=2,
        unload_text_encoder=False,
    )
    t.data_loader = None
    t.data_loader_reg = None
    t.epoch_num = 0
    t.sample_config = SimpleNamespace(samples=[SimpleNamespace(prompt="old")])
    t.is_caching_text_embeddings = False
    t.get_runtime_batch_size = lambda: 2
    t.get_runtime_prompts = lambda: ["new prompt"]
    t.get_runtime_scale_lr_mean = lambda: None
    t.get_runtime_scale_lr_std = lambda: None
    t._read_runtime_scale_lr_mask = lambda: RuntimeScaleLrMaskRead("absent")
    t.get_runtime_scale_lr_by_index = lambda: False

    t.apply_runtime_batch_size()
    t.apply_runtime_prompts()
    t.apply_runtime_scale_lr()

    assert calls == []
    assert opt.get_learning_rates()[1] == pytest.approx(lr)
    assert g1.get("warmup_active") is False
    assert "warmup_start" not in g1
    assert {k: g1[k] for k in g1 if k.startswith("warmup_")} == warmup_keys

    t.get_runtime_scale_lr_mean = lambda: 0.0
    t.get_runtime_scale_lr_std = lambda: 0.5
    t.get_runtime_scale_lr_by_index = lambda: True
    t.apply_runtime_scale_lr()
    assert calls == ["config", "by_index"]
    assert g1["warmup_lr_previous"] == pytest.approx(prev)
    assert opt.get_learning_rates()[1] == pytest.approx(lr)

    _step(opt, params)
    eps1 = float(g1["eps"][1])
    assert g1["warmup_start"] == pytest.approx(prev)
    assert g1["warmup_lr"] == pytest.approx(prev)
    assert g1["warmup_lr"] != pytest.approx(lr * eps1)

    target = opt._to_effective_lr(lr, g1)
    _step(opt, params)
    assert g1["warmup_lr"] != pytest.approx(lr * eps1)
    assert g1["warmup_lr"] < prev
    assert g1["warmup_lr"] > target
