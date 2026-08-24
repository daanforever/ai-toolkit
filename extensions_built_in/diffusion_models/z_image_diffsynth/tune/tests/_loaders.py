"""File-path loaders for tune modules (avoid parent package DiT import)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_TUNE_DIR = Path(__file__).resolve().parents[1]
_PKG = "_zids_tune_test"


def _exec(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_overlay():
    return _exec(f"{_PKG}.overlay", _TUNE_DIR / "overlay.py")


def load_probe():
    return _exec(f"{_PKG}.probe", _TUNE_DIR / "probe.py")


def load_rubric():
    return _exec(f"{_PKG}.rubric", _TUNE_DIR / "rubric.py")


def load_driver():
    """Load overlay/probe/rubric + __main__.py as a package so relative imports work."""
    if _PKG in sys.modules and hasattr(sys.modules[_PKG], "driver"):
        return sys.modules[_PKG].driver

    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(_TUNE_DIR)]
    pkg.__file__ = str(_TUNE_DIR / "__init__.py")
    sys.modules[_PKG] = pkg

    for leaf in ("overlay", "probe", "rubric"):
        mod = _exec(f"{_PKG}.{leaf}", _TUNE_DIR / f"{leaf}.py")
        setattr(pkg, leaf, mod)

    driver = _exec(f"{_PKG}.driver", _TUNE_DIR / "__main__.py")
    pkg.driver = driver
    return driver


def recipe_path() -> Path:
    return _TUNE_DIR / "config.yaml"
