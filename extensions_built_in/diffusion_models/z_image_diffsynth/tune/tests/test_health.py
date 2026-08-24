"""health_from_tb."""

from __future__ import annotations

from pathlib import Path

import pytest

from ._loaders import load_rubric


@pytest.fixture(scope="module")
def rubric():
    return load_rubric()


def _write_loss_tb(log_dir: Path, series: list[tuple[int, float]], tag: str = "loss") -> None:
    from torch.utils.tensorboard import SummaryWriter

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    for step, value in series:
        writer.add_scalar(tag, value, global_step=step)
    writer.flush()
    writer.close()


def _write_tb_scalars(
    log_dir: Path, tag_series: dict[str, list[tuple[int, float]]]
) -> None:
    from torch.utils.tensorboard import SummaryWriter

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    for tag, series in tag_series.items():
        for step, value in series:
            writer.add_scalar(tag, value, global_step=step)
    writer.flush()
    writer.close()


def test_health_missing_dir_no_tb(rubric, tmp_path):
    missing = tmp_path / "no_such_tb"
    result = rubric.health_from_tb(missing, warmup_steps=8, instability_max=1.0)
    assert result.ok is False
    assert result.reason == "no_tb"


def test_health_in_warmup_only_skips_loss_ratio(rubric, tmp_path):
    log_dir = tmp_path / "tb_warmup"
    # All steps strictly below warmup → post-warmup empty → skip loss_ratio → ok
    _write_loss_tb(log_dir, [(0, 1.0), (3, 2.0), (7, 9.0)])
    result = rubric.health_from_tb(log_dir, warmup_steps=8, instability_max=1.0)
    assert result.ok is True
    assert result.reason is None


def test_health_post_warmup_8x_loss_ratio_fails(rubric, tmp_path):
    log_dir = tmp_path / "tb_ratio"
    # Post-warmup (step >= 8): first 20% low, last 20% > 8x
    series = [(8, 1.0), (9, 1.0), (10, 1.0), (11, 1.0), (12, 1.0)]
    series += [(20, 10.0), (21, 10.0), (22, 10.0), (23, 10.0), (24, 10.0)]
    _write_loss_tb(log_dir, series)
    result = rubric.health_from_tb(log_dir, warmup_steps=8, instability_max=1.0)
    assert result.ok is False
    assert result.reason == "loss_ratio"


def test_health_flat_update_rms_fails(rubric, tmp_path):
    log_dir = tmp_path / "tb_flat_update"
    rms = [(8, 0.0), (9, 1e-12), (10, 0.0), (11, 0.0)]
    _write_loss_tb(log_dir, rms, tag="train/update_rms")
    result = rubric.health_from_tb(log_dir, warmup_steps=8, instability_max=1.0)
    assert result.ok is False
    assert result.reason == "flat_update"


def test_health_nonzero_update_rms_ok(rubric, tmp_path):
    log_dir = tmp_path / "tb_healthy_rms"
    loss = [(8, 1.0), (9, 1.0), (10, 1.0), (11, 1.0)]
    rms = [(8, 1e-3), (9, 1e-3), (10, 1e-3), (11, 1e-3)]
    _write_tb_scalars(
        log_dir,
        {"loss": loss, "train/update_rms": rms},
    )
    result = rubric.health_from_tb(log_dir, warmup_steps=8, instability_max=1.0)
    assert result.ok is True
    assert result.reason is None


def test_health_flat_grad_rms_fallback_fails(rubric, tmp_path):
    log_dir = tmp_path / "tb_flat_grad"
    rms = [(8, 0.0), (9, 0.0), (10, 0.0), (11, 0.0)]
    _write_loss_tb(log_dir, rms, tag="train/grad_rms")
    result = rubric.health_from_tb(log_dir, warmup_steps=8, instability_max=1.0)
    assert result.ok is False
    assert result.reason == "flat_update"
