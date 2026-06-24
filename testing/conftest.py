"""Shared pytest fixtures for integration tests under testing/."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _skip_unless_real_zimage_stack() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from testing.test_flowmatch_snr_real_model import _resolve_model_paths

    model_path, _ = _resolve_model_paths()
    if not model_path or not os.path.isdir(model_path):
        pytest.skip(
            "Z-Image model path missing "
            "(set ZIMAGE_DIFFSYNTH_MODEL_PATH or DEFAULT_ZIMAGE_MODEL_PATH)"
        )
    dataset_dir = REPO_ROOT / "temp" / "test_train"
    if not dataset_dir.is_dir() or not list(dataset_dir.glob("*.png")):
        pytest.skip(f"Dataset missing or empty: {dataset_dir}")


@pytest.fixture(scope="module")
def snr_probe_records(tmp_path_factory):
    """
    One run_job / one model load for all tests in test_flowmatch_snr_real_model.py.
    """
    _skip_unless_real_zimage_stack()

    from testing.test_flowmatch_snr_real_model import (
        SNR_REAL_MODEL_PROBES,
        ProbeRunResult,
        _run_probes_in_one_job,
    )

    work_root = tmp_path_factory.mktemp("flowmatch_snr")
    result: ProbeRunResult = _run_probes_in_one_job(work_root, SNR_REAL_MODEL_PROBES)
    yield result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
