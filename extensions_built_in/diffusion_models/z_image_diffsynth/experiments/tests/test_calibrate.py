"""calibrate: equivalence, exchange rates, beta* interpolation."""

from __future__ import annotations

from ._loaders import load_calibrate


def test_classify_equivalent():
    cal = load_calibrate()
    out = cal.classify_equivalence(
        equiv_ratio=1.0, equiv_cosine=1.0, rel_tol=0.25, cosine_min=0.9
    )
    assert out["status"] == "equivalent"


def test_classify_partial_ratio4_cosine1():
    """Δ_lr = 4·Δ_continue and Δ_b2 = Δ_continue → ratio=4, cosine=1 → partial."""
    cal = load_calibrate()
    out = cal.classify_equivalence(
        equiv_ratio=4.0, equiv_cosine=1.0, rel_tol=0.25, cosine_min=0.9
    )
    assert out["status"] == "partial"


def test_classify_divergent():
    cal = load_calibrate()
    out = cal.classify_equivalence(
        equiv_ratio=4.0, equiv_cosine=0.0, rel_tol=0.25, cosine_min=0.9
    )
    assert out["status"] == "divergent"


def test_pick_beta_star_nearest():
    cal = load_calibrate()
    out = cal.pick_beta_star(
        rms_lr_x4=4.0,
        rms_by_beta={0.7: 1.1, 0.8: 1.5, 0.95: 3.8, 0.99: 2.0},
    )
    assert out["beta_star"] == 0.95
    assert out.get("reason") is None
    assert abs(out["r_star"] - 3.8 / 4.0) < 1e-9


def test_pick_beta_star_empty_grid():
    cal = load_calibrate()
    out = cal.pick_beta_star(rms_lr_x4=4.0, rms_by_beta={})
    assert out["beta_star"] is None
    assert out["reason"] == "empty_grid"


def test_exchange_rates():
    cal = load_calibrate()
    out = cal.exchange_rates(
        rms_continue=1.0, rms_lr_x4=4.0, rms_beta2_lo=1.0
    )
    assert abs(out["s_lr"] - 4.0) < 1e-9
    assert abs(out["s_b2"] - 1.0) < 1e-9


def test_stationary_v_diagnostic_true():
    cal = load_calibrate()
    assert (
        cal.stationary_v_diagnostic(
            ratios_vs_continue={0.7: 1.02, 0.8: 0.98, 0.95: 1.01},
            s_lr=4.0,
        )
        is True
    )


def test_stationary_v_diagnostic_false_low_s_lr():
    cal = load_calibrate()
    assert (
        cal.stationary_v_diagnostic(
            ratios_vs_continue={0.7: 1.02, 0.8: 0.98},
            s_lr=1.5,
        )
        is False
    )


def test_stationary_v_diagnostic_false_spread():
    cal = load_calibrate()
    assert (
        cal.stationary_v_diagnostic(
            ratios_vs_continue={0.7: 0.5, 0.8: 2.0},
            s_lr=4.0,
        )
        is False
    )
