"""Resume seek for cosine / cosine_with_restarts + warmup LR scheduler."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from toolkit.scheduler import get_lr_scheduler

STEPS = 200
WARMUP_STEPS = STEPS // 2  # 100
T_0 = STEPS - WARMUP_STEPS  # 100
LR = 4e-5
ETA_MIN = 1e-5

SCHEDULER_NAMES = ("cosine", "cosine_with_restarts")


def _make(scheduler_name: str, resume_step: int = 0, **extra):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    kwargs = dict(
        resume_step=resume_step,
        warmup_steps=WARMUP_STEPS,
        total_iters=STEPS,
        eta_min=ETA_MIN,
    )
    kwargs.update(extra)
    scheduler = get_lr_scheduler(scheduler_name, optimizer, **kwargs)
    return optimizer, scheduler


def _snapshot(optimizer, scheduler):
    warm, cosine = scheduler._schedulers
    snap = {
        "lr": optimizer.param_groups[0]["lr"],
        "seq_last": scheduler.last_epoch,
        "warm_last": warm.last_epoch,
        "cosine_last": cosine.last_epoch,
    }
    if hasattr(cosine, "T_cur"):
        snap["T_cur"] = cosine.T_cur
        snap["T_i"] = cosine.T_i
    return snap


def uninterrupted_at(scheduler_name: str, S: int, **extra):
    """Start of training step S via stepping from resume_step=0."""
    optimizer, scheduler = _make(scheduler_name, resume_step=0, **extra)
    for _ in range(S):
        scheduler.step()
    return optimizer, scheduler, _snapshot(optimizer, scheduler)


def resumed_at(scheduler_name: str, S: int, **extra):
    optimizer, scheduler = _make(scheduler_name, resume_step=S, **extra)
    return optimizer, scheduler, _snapshot(optimizer, scheduler)


@pytest.mark.parametrize("scheduler_name", SCHEDULER_NAMES)
def test_seek_matches_uninterrupted(scheduler_name):
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
        _, ref_sched, ref = uninterrupted_at(scheduler_name, S)
        _, resume_sched, got = resumed_at(scheduler_name, S)

        assert got["lr"] == pytest.approx(ref["lr"]), f"{scheduler_name} S={S} lr"
        assert got["seq_last"] == ref["seq_last"], f"{scheduler_name} S={S} seq_last"
        assert got["warm_last"] == ref["warm_last"], f"{scheduler_name} S={S} warm_last"
        assert got["cosine_last"] == ref["cosine_last"], f"{scheduler_name} S={S} cosine_last"
        if "T_cur" in ref:
            assert got["T_cur"] == ref["T_cur"], f"{scheduler_name} S={S} T_cur"
            assert got["T_i"] == ref["T_i"], f"{scheduler_name} S={S} T_i"

        if S + 1 >= WARMUP_STEPS:
            assert got["warm_last"] == WARMUP_STEPS - 1, f"{scheduler_name} S={S} inactive warm"

        ref_sched.step()
        resume_sched.step()
        assert resume_sched.optimizer.param_groups[0]["lr"] == pytest.approx(
            ref_sched.optimizer.param_groups[0]["lr"]
        ), f"{scheduler_name} S={S} after +1 step"


def test_phase_boundary_lrs_cosine():
    boundaries = {
        "lambda_start": 0,
        "lambda_end": WARMUP_STEPS - 2,
        "cosine_start": WARMUP_STEPS - 1,
        "cosine_end": STEPS - 1,
    }

    _, _, snap0 = uninterrupted_at("cosine", boundaries["lambda_start"])
    _, _, snap_peak = uninterrupted_at("cosine", boundaries["lambda_end"])
    _, _, snap_cos0 = uninterrupted_at("cosine", boundaries["cosine_start"])
    _, _, snap_end = uninterrupted_at("cosine", boundaries["cosine_end"])

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
        _, _, ref = uninterrupted_at("cosine", S)
        _, _, got = resumed_at("cosine", S)
        assert got["lr"] == pytest.approx(ref["lr"]), name
        assert got["seq_last"] == ref["seq_last"], name
        assert got["cosine_last"] == ref["cosine_last"], name


def test_phase_boundary_lrs_cosine_with_restarts():
    """WarmRestarts: at steps-1 cycle wraps to peak; near-eta_min is T_cur == T_0-1."""
    near_eta_min_S = WARMUP_STEPS + T_0 - 2  # 198 → T_cur == T_0 - 1
    boundaries = {
        "lambda_start": 0,
        "lambda_end": WARMUP_STEPS - 2,
        "cosine_start": WARMUP_STEPS - 1,
        "near_eta_min": near_eta_min_S,
    }

    _, _, snap0 = uninterrupted_at("cosine_with_restarts", boundaries["lambda_start"])
    _, _, snap_peak = uninterrupted_at("cosine_with_restarts", boundaries["lambda_end"])
    _, _, snap_cos0 = uninterrupted_at("cosine_with_restarts", boundaries["cosine_start"])
    _, _, snap_near = uninterrupted_at("cosine_with_restarts", boundaries["near_eta_min"])

    assert snap0["lr"] == pytest.approx(ETA_MIN, abs=ETA_MIN * 0.05)
    assert snap_peak["lr"] == pytest.approx(LR)
    assert snap_cos0["lr"] == pytest.approx(LR, abs=LR * 1e-4)
    assert snap_cos0["cosine_last"] == 0
    assert snap_cos0["T_cur"] == 0

    assert snap_near["T_cur"] == T_0 - 1
    assert snap_near["lr"] == pytest.approx(ETA_MIN, abs=ETA_MIN * 0.05)

    for name, S in boundaries.items():
        _, _, ref = uninterrupted_at("cosine_with_restarts", S)
        _, _, got = resumed_at("cosine_with_restarts", S)
        assert got["lr"] == pytest.approx(ref["lr"]), name
        assert got["seq_last"] == ref["seq_last"], name
        assert got["T_cur"] == ref["T_cur"], name
        assert got["T_i"] == ref["T_i"], name


@pytest.mark.parametrize("scheduler_name", SCHEDULER_NAMES)
def test_mid_warmup_interrupt(scheduler_name):
    S = WARMUP_STEPS // 2  # steps // 4

    ref_opt, ref_sched, ref = uninterrupted_at(scheduler_name, S)
    resume_opt, resume_sched, got = resumed_at(scheduler_name, S)

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
        ), f"{scheduler_name} catch-up at S={target}"


def test_cosine_with_restarts_seek_after_restart_t_mult_2():
    """Past first T_0 in main phase with T_mult=2: T_i doubles after restart."""
    # After switch, first cycle length T_0=100. Past restart: main epoch > T_0.
    # S such that cosine_epoch = (S+1) - warmup > T_0, e.g. cosine_epoch=110 → S=209
    # but STEPS=200 limits main to 100 epochs from resume_step=0 path within STEPS.
    # Use larger total_iters so we can pass the first restart.
    steps = 400
    warmup = steps // 2  # 200
    t0 = steps - warmup  # 200 with default T_0 from total_iters-warmup... wait
    # With T_0 explicit smaller than main length so restart happens inside run:
    t0 = 50
    # main length = steps - warmup = 200; restart at main epoch 50.
    # Past restart: cosine_epoch=60 → seq=warmup+60=260 → S=259
    S = warmup + 60 - 1  # 259

    extra = dict(
        warmup_steps=warmup,
        total_iters=steps,
        T_0=t0,
        T_mult=2,
        eta_min=ETA_MIN,
    )
    # _make merges total_iters/warmup via kwargs — pass through resumed/uninterrupted
    model = nn.Linear(1, 1)
    opt_ref = torch.optim.SGD(model.parameters(), lr=LR)
    ref_sched = get_lr_scheduler(
        "cosine_with_restarts",
        opt_ref,
        resume_step=0,
        **extra,
    )
    for _ in range(S):
        ref_sched.step()
    ref = _snapshot(opt_ref, ref_sched)

    model2 = nn.Linear(1, 1)
    opt_got = torch.optim.SGD(model2.parameters(), lr=LR)
    got_sched = get_lr_scheduler(
        "cosine_with_restarts",
        opt_got,
        resume_step=S,
        **extra,
    )
    got = _snapshot(opt_got, got_sched)

    assert got["lr"] == pytest.approx(ref["lr"])
    assert got["seq_last"] == ref["seq_last"]
    assert got["cosine_last"] == ref["cosine_last"]
    assert got["T_cur"] == ref["T_cur"]
    assert got["T_i"] == ref["T_i"]
    # After first restart with T_mult=2: T_i should be 2 * T_0
    assert got["T_i"] == t0 * 2
    assert got["T_cur"] == 60 - t0  # epoch 60 into main → T_cur=10 after restart at 50


def test_constant_accepts_resume_step_kwarg():
    """resume_step is toolkit-internal; must not be forwarded to ConstantLR."""
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = get_lr_scheduler(
        "constant",
        optimizer,
        resume_step=10,
        total_iters=100,
        factor=1.0,
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)
