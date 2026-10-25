"""Opt-in wrapper. The repro must start in a fresh interpreter.

`testing/conftest.py` imports torch, so collecting this file cannot set
`PYTORCH_CUDA_ALLOC_CONF` for the training process. The child is `repro.py`.

  OOM_REPRO=1 venv\\Scripts\\python.exe -m pytest testing/oom/test_repro.py -q -s
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OOM_REPRO") != "1",
    reason="set OOM_REPRO=1 to run the CUDA memory-cap LoRA repro",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_oom_repro_runs():
    script = Path(__file__).with_name("repro.py")
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO_ROOT,
        env=os.environ.copy(),
        check=False,
    )
    assert completed.returncode == 0, f"oom repro exited {completed.returncode}"
