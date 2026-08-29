"""File-path loaders for experiments modules (avoid parent DiT import)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parents[1]
_PKG = "_zids_exp_test"


def _exec(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_overlay():
    return _exec(f"{_PKG}.overlay", _EXP_DIR / "overlay.py")


def load_probe():
    return _exec(f"{_PKG}.probe", _EXP_DIR / "probe.py")


def load_capture():
    return _exec(f"{_PKG}.capture", _EXP_DIR / "capture.py")


def load_calibrate():
    return _exec(f"{_PKG}.calibrate", _EXP_DIR / "calibrate.py")


def load_session():
    return _exec(f"{_PKG}.session", _EXP_DIR / "session.py")


def load_driver():
    if _PKG in sys.modules and hasattr(sys.modules[_PKG], "driver"):
        return sys.modules[_PKG].driver

    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(_EXP_DIR)]
    pkg.__file__ = str(_EXP_DIR / "__init__.py")
    sys.modules[_PKG] = pkg

    for leaf in ("overlay", "probe", "capture", "calibrate"):
        mod = _exec(f"{_PKG}.{leaf}", _EXP_DIR / f"{leaf}.py")
        setattr(pkg, leaf, mod)

    cases_pkg = types.ModuleType(f"{_PKG}.cases")
    cases_pkg.__path__ = [str(_EXP_DIR / "cases")]
    sys.modules[f"{_PKG}.cases"] = cases_pkg
    pkg.cases = cases_pkg

    driver = _exec(f"{_PKG}.driver", _EXP_DIR / "__main__.py")
    pkg.driver = driver
    return driver


def recipe_path() -> Path:
    return _EXP_DIR / "config.yaml"
