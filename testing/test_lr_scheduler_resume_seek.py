"""Resume seek for cosine + warmup LR scheduler."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from toolkit.scheduler import get_lr_scheduler

STEPS = 200
WARMUP_STEPS = STEPS // 2  # 100
LR = 4e-5
ETA_MIN = 1e-5


def _make(resume_step: int = 0):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = get_lr_scheduler(
        "cosine",
        optimizer,
        resume_step=resume_step,
        warmup_steps=WARMUP_STEPS,
        total_iters=STEPS,
        eta_min=ETA_MIN,
    )
    return optimizer, scheduler


def _snapshot(optimizer, scheduler):
    warm, cosine = scheduler._schedulers
    return {
        "lr": optimizer.param_groups[0]["lr"],
        "seq_last": scheduler.last_epoch,
        "warm_last": warm.last_epoch,
        "cosine_last": cosine.last_epoch,
    }


def uninterrupted_at(S: int):
    """Start of training step S via stepping from resume_step=0."""
    optimizer, scheduler = _make(resume_step=0)
    for _ in range(S):
        scheduler.step()
    return optimizer, scheduler, _snapshot(optimizer, scheduler)


def resumed_at(S: int):
    optimizer, scheduler = _make(resume_step=S)
    return optimizer, scheduler, _snapshot(optimizer, scheduler)


def test_seek_matches_uninterrupted():
    steps_to_check = {
        0,
        1,
        WARMUP_STEPS // 2,
        WARMUP_STEPS - 2,
        WARMUP_STEPS - 1,
        WARMUP_STEPS,
        WARMUP_STEPS + 1,
        WARMUP_STEPS + STEPS // 4,
    }
    for S in sorted(steps_to_check):
        _, ref_sched, ref = uninterrupted_at(S)
        _, resume_sched, got = resumed_at(S)

        assert got["lr"] == pytest.approx(ref["lr"]), f"S={S} lr"
        assert got["seq_last"] == ref["seq_last"], f"S={S} seq_last"
        assert got["warm_last"] == ref["warm_last"], f"S={S} warm_last"
        assert got["cosine_last"] == ref["cosine_last"], f"S={S} cosine_last"

        if S + 1 >= WARMUP_STEPS:
            assert got["warm_last"] == WARMUP_STEPS - 1, f"S={S} inactive warm"

        ref_sched.step()
        resume_sched.step()
        assert resume_sched.optimizer.param_groups[0]["lr"] == pytest.approx(
            ref_sched.optimizer.param_groups[0]["lr"]
        ), f"S={S} after +1 step"


def test_phase_boundary_lrs():
    boundaries = {
        "lambda_start": 0,
        "lambda_end": WARMUP_STEPS - 2,
        "cosine_start": WARMUP_STEPS - 1,
        "cosine_end": STEPS - 1,
    }

    _, _, snap0 = uninterrupted_at(boundaries["lambda_start"])
    _, _, snap_peak = uninterrupted_at(boundaries["lambda_end"])
    _, _, snap_cos0 = uninterrupted_at(boundaries["cosine_start"])
    _, _, snap_end = uninterrupted_at(boundaries["cosine_end"])

    assert snap0["lr"] == pytest.approx(ETA_MIN, abs=ETA_MIN * 0.05)
    assert snap_peak["lr"] == pytest.approx(LR)
    assert snap_peak["warm_last"] == WARMUP_STEPS - 1
    assert snap_peak["seq_last"] == WARMUP_STEPS - 1

    assert snap_cos0["lr"] == pytest.approx(LR, abs=LR * 1e-4)
    assert snap_cos0["cosine_last"] == 0
    assert snap_cos0["seq_last"] == WARMUP_STEPS

    assert snap_end["lr"] == pytest.approx(ETA_MIN, abs=ETA_MIN * 1e-4)
    assert snap_end["cosine_last"] == STEPS - WARMUP_STEPS
    assert snap_end["seq_last"] == STEPS

    for name, S in boundaries.items():
        _, _, ref = uninterrupted_at(S)
        _, _, got = resumed_at(S)
        assert got["lr"] == pytest.approx(ref["lr"]), name
        assert got["seq_last"] == ref["seq_last"], name
        assert got["cosine_last"] == ref["cosine_last"], name


def test_mid_warmup_interrupt():
    S = WARMUP_STEPS // 2  # steps // 4

    ref_opt, ref_sched, ref = uninterrupted_at(S)
    resume_opt, resume_sched, got = resumed_at(S)

    assert got["lr"] == pytest.approx(ref["lr"])
    assert ETA_MIN < got["lr"] < LR

    targets = [WARMUP_STEPS - 2, WARMUP_STEPS - 1, STEPS - 1]
    cur = S
    for target in targets:
        while cur < target:
            ref_sched.step()
            resume_sched.step()
            cur += 1
        assert resume_opt.param_groups[0]["lr"] == pytest.approx(
            ref_opt.param_groups[0]["lr"]
        ), f"catch-up at S={target}"
