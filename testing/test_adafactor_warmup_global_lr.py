"""Unit tests for Adafactor group-level warmup (_global_lr / _warmup_update_group)."""

import pytest
import torch

from toolkit.optimizers.adafactor import Adafactor


def test_warmup_single_increment_per_step_two_params():
    """One optimizer.step must advance group warmup once, not once per parameter."""
    w1 = torch.nn.Parameter(torch.ones(4))
    w2 = torch.nn.Parameter(torch.ones(4))
    opt = Adafactor(
        [w1, w2],
        lr=1.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=10,
        beta1=None,
        weight_decay=0.0,
    )
    w1.grad = torch.zeros_like(w1)
    w2.grad = torch.zeros_like(w2)
    g = opt.param_groups[0]
    eps1 = g["eps"][1]
    opt.step()
    assert g["warmup_progress"] == 1
    assert g["warmup_lr"] == pytest.approx(1.0 * eps1)


def test_warmup_advances_when_no_grad():
    """Warmup ticks once per step() even if no parameter receives an update."""
    w1 = torch.nn.Parameter(torch.ones(2))
    opt = Adafactor(
        [w1],
        lr=0.5,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=5,
        beta1=None,
        weight_decay=0.0,
    )
    g = opt.param_groups[0]
    eps1 = g["eps"][1]
    w1.grad = None
    opt.step()
    assert g["warmup_progress"] == 1
    assert g["warmup_lr"] == pytest.approx(0.5 * eps1)


def test_warmup_no_spurious_segment_after_complete():
    """After warmup finishes, idle steps must not restart a segment (warmup_target vs lr)."""
    w1 = torch.nn.Parameter(torch.ones(1))
    opt = Adafactor(
        [w1],
        lr=1.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=2,
        beta1=None,
        weight_decay=0.0,
    )
    w1.grad = torch.zeros_like(w1)
    g = opt.param_groups[0]
    opt.step()
    opt.step()
    assert not g.get("warmup_active", True)
    assert g["warmup_target"] == 1.0
    opt.step()
    assert not g.get("warmup_active", True)
    assert "warmup_progress" not in g


def test_warmup_final_intermediate_lr_is_applied_before_cleanup():
    """For warmup_steps=2, step2 must still use intermediate warmup_lr, not jump to target."""
    w1 = torch.nn.Parameter(torch.ones(1))
    opt = Adafactor(
        [w1],
        lr=1.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=2,
        beta1=None,
        weight_decay=0.0,
        factored=False,
    )
    g = opt.param_groups[0]
    eps1 = g["eps"][1]
    expected_step_1_lr = 1.0 * eps1
    expected_step_2_lr = expected_step_1_lr + (1.0 - expected_step_1_lr) / 2.0

    w1.grad = torch.zeros_like(w1)
    opt.step()
    assert g["lr_mean"].item() == pytest.approx(expected_step_1_lr)

    w1.grad = torch.zeros_like(w1)
    opt.step()
    assert g["lr_mean"].item() == pytest.approx(expected_step_2_lr)

    # Cleanup happens on the next step, then base lr is used.
    w1.grad = torch.zeros_like(w1)
    opt.step()
    assert g["lr_mean"].item() == pytest.approx(1.0)


def test_warmup_groups_complete_independently():
    """When one group completes its segment, other groups keep warming up."""
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    opt = Adafactor(
        [{"params": [p1]}, {"params": [p2]}],
        lr=1.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=10,
        beta1=None,
        weight_decay=0.0,
    )
    g0, g1 = opt.param_groups
    g0["warmup_steps"] = 2
    g1["warmup_steps"] = 10

    for _ in range(2):
        p1.grad = torch.zeros_like(p1)
        p2.grad = torch.zeros_like(p2)
        opt.step()

    # First group completed its segment; delayed cleanup still pending.
    assert g0.get("warmup_complete_pending_cleanup") is True
    assert "warmup_lr" in g0

    # Second group must still be warming independently.
    assert g1.get("warmup_active") is True
    assert "warmup_lr" in g1
    assert g1["warmup_progress"] == 2


def test_warmup_boost_up_overshoots_then_snaps_to_target():
    """Upward segment ramps toward lr * boost; cleanup snaps to real lr."""
    w1 = torch.nn.Parameter(torch.ones(1))
    lr = 1e-4
    boost = 2.0
    steps = 4
    opt = Adafactor(
        [w1],
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=steps,
        warmup_boost=boost,
        beta1=None,
        weight_decay=0.0,
        factored=False,
    )
    g = opt.param_groups[0]
    eps1 = g["eps"][1]
    lr_start = lr * eps1
    lr_interp = lr * boost
    expected_delta = (lr_interp - lr_start) / steps

    w1.grad = torch.zeros_like(w1)
    opt.step()
    assert g["warmup_target"] == lr
    assert g["warmup_delta"] == pytest.approx(expected_delta)
    assert g["warmup_lr"] == pytest.approx(lr_start)

    for _ in range(steps - 1):
        w1.grad = torch.zeros_like(w1)
        opt.step()

    assert g.get("warmup_complete_pending_cleanup") is True
    last_warmup_lr = lr_start + (steps - 1) * expected_delta
    assert g["warmup_lr"] == pytest.approx(last_warmup_lr)
    assert last_warmup_lr > lr

    w1.grad = torch.zeros_like(w1)
    opt.step()
    assert "warmup_lr" not in g
    assert g["lr_mean"].item() == pytest.approx(lr)
    assert g["warmup_lr_previous"] == pytest.approx(lr)


def test_warmup_boost_down_undershoots_then_snaps_to_target():
    """Downward segment ramps toward lr / boost; cleanup snaps to real lr."""
    w1 = torch.nn.Parameter(torch.ones(1))
    lr_high = 1e-4
    lr_low = 1e-5
    boost = 2.0
    # Need enough steps so last intermediate (p=N-1) crosses below real lr_low.
    steps = 20
    opt = Adafactor(
        [w1],
        lr=lr_high,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=steps,
        warmup_boost=boost,
        beta1=None,
        weight_decay=0.0,
        factored=False,
    )
    g = opt.param_groups[0]

    # Complete initial upward segment and cleanup so previous == lr_high.
    for _ in range(steps + 1):
        w1.grad = torch.zeros_like(w1)
        opt.step()
    assert g["warmup_lr_previous"] == pytest.approx(lr_high)

    g["lr"] = lr_low
    lr_interp = lr_low / boost
    expected_delta = (lr_interp - lr_high) / steps

    w1.grad = torch.zeros_like(w1)
    opt.step()
    assert g["warmup_target"] == lr_low
    assert g["warmup_delta"] == pytest.approx(expected_delta)
    assert g["warmup_lr"] == pytest.approx(lr_high)

    for _ in range(steps - 1):
        w1.grad = torch.zeros_like(w1)
        opt.step()

    assert g.get("warmup_complete_pending_cleanup") is True
    last_warmup_lr = lr_high + (steps - 1) * expected_delta
    assert g["warmup_lr"] == pytest.approx(last_warmup_lr)
    assert last_warmup_lr < lr_low

    w1.grad = torch.zeros_like(w1)
    opt.step()
    assert "warmup_lr" not in g
    assert g["lr_mean"].item() == pytest.approx(lr_low)
    assert g["warmup_lr_previous"] == pytest.approx(lr_low)


def test_warmup_boost_independent_groups_keep_previous_on_own_cleanup():
    """Each group snaps warmup_lr_previous to its real lr only when it completes."""
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    lr = 1.0
    boost = 2.0
    opt = Adafactor(
        [{"params": [p1]}, {"params": [p2]}],
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=10,
        warmup_boost=boost,
        beta1=None,
        weight_decay=0.0,
    )
    g0, g1 = opt.param_groups
    g0["warmup_steps"] = 2
    g1["warmup_steps"] = 10

    for _ in range(2):
        p1.grad = torch.zeros_like(p1)
        p2.grad = torch.zeros_like(p2)
        opt.step()

    assert g0.get("warmup_complete_pending_cleanup") is True
    assert g1.get("warmup_active") is True
    assert "warmup_lr" in g1
    # Still mid-ramp toward lr * boost; previous tracks intermediate warmup_lr.
    assert g1["warmup_lr_previous"] == pytest.approx(g1["warmup_lr"])
    assert g1["warmup_lr"] > lr * g1["eps"][1]

    p1.grad = torch.zeros_like(p1)
    p2.grad = torch.zeros_like(p2)
    opt.step()
    assert "warmup_lr" not in g0
    assert g0["warmup_lr_previous"] == pytest.approx(lr)
    assert g1.get("warmup_active") is True


def test_warmup_scale_lr_lower_index_scale_finishes_earlier():
    """With scale_lr_by_index, lower effective-target groups complete warmup first."""
    p_hi = torch.nn.Parameter(torch.ones(2))
    p_lo = torch.nn.Parameter(torch.ones(2))
    p_max = torch.nn.Parameter(torch.ones(2))
    lr = 1.0
    steps = 10
    opt = Adafactor(
        [
            {"params": [p_hi], "index": 0},
            {"params": [p_lo], "index": 1},
            {"params": [p_max], "index": 2},
        ],
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=True,
        warmup_steps=steps,
        beta1=None,
        weight_decay=0.0,
        factored=False,
        scale_lr_by_index=True,
        scale_lr_factor=1.0,
    )
    g_hi, g_lo, g_max = opt.param_groups
    eps0 = float(g_hi["eps"][0])
    eps1 = float(g_hi["eps"][1])
    assert opt._index_lr_multiplier(g_hi) == pytest.approx(1.0)
    assert opt._index_lr_multiplier(g_lo) == pytest.approx(0.5)
    assert opt._index_lr_multiplier(g_max) == pytest.approx(0.0)

    target_lo = opt._to_effective_lr(lr, g_lo)
    assert target_lo == pytest.approx(lr * 0.5 + eps0)

    lr_start_u = lr * eps1
    delta_ref = (lr - lr_start_u) / steps
    start_lo = opt._to_effective_lr(lr_start_u, g_lo)
    interp_lo = opt._to_effective_lr(lr, g_lo)
    steps_to_lo = 0
    for p in range(1, steps + 1):
        if start_lo + (p - 1) * delta_ref >= interp_lo:
            steps_to_lo = p
            break
    assert 0 < steps_to_lo < steps

    for _ in range(steps_to_lo):
        p_hi.grad = torch.zeros_like(p_hi)
        p_lo.grad = torch.zeros_like(p_lo)
        p_max.grad = torch.zeros_like(p_max)
        opt.step()

    assert g_lo.get("warmup_complete_pending_cleanup") is True
    assert g_hi.get("warmup_active") is True
    assert "warmup_lr" in g_hi

    p_hi.grad = torch.zeros_like(p_hi)
    p_lo.grad = torch.zeros_like(p_lo)
    p_max.grad = torch.zeros_like(p_max)
    opt.step()

    assert "warmup_lr" not in g_lo
    assert g_lo["lr_mean"].item() == pytest.approx(target_lo)
    assert g_hi.get("warmup_active") is True or "warmup_lr" in g_hi
