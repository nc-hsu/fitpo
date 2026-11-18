"""Tests for fit_cbf_knots_backbone.

Covers both ways of placing the knots (force-fraction snapping and explicit
knot displacements), the residual-plateau mode, and the drop-to-zero mode.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from fitpo import fit_cbf_knots_backbone
from fitpo._cbf import _snap_descending_crossing
from fitpo._common import _interval_area_error

CURVES_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tetralinear(
    d_y=20.0, f_y=1000.0,
    d_f=100.0, f_f=900.0,
    d_r=120.0, f_r=150.0,
    d_max=200.0,
    n=401,
) -> np.ndarray:
    """Noise-free tetralinear curve sampled on a uniform displacement grid."""
    xs = np.asarray([0.0, d_y, d_f, d_r, d_max])
    ys = np.asarray([0.0, f_y, f_f, f_r, f_r])
    d = np.linspace(0.0, d_max, n)
    return np.column_stack([d, np.interp(d, xs, ys)])


# ---------------------------------------------------------------------------
# The force-fraction path
# ---------------------------------------------------------------------------


def test_returns_ordered_breakpoints():
    # The force never drops below 5 % of peak, so d_r snaps to the last point
    # and the zero-length plateau row is dropped: 4 rows.
    with pytest.warns(UserWarning, match="never falls to residual_force_fraction"):
        bb = fit_cbf_knots_backbone(_tetralinear()).fitted_po
    assert bb.shape == (4, 2)
    assert bb[0].tolist() == [0.0, 0.0]
    assert (np.diff(bb[:, 0]) > 0).all()
    # With a residual crossing inside the data, the plateau row is kept.
    bb = fit_cbf_knots_backbone(_tetralinear(), residual_force_fraction=0.20).fitted_po
    assert bb.shape == (5, 2)
    assert (np.diff(bb[:, 0]) > 0).all()


def test_knots_land_on_the_requested_force_fractions():
    # d_f and d_r are snapped to the descending crossings of the fractions,
    # so the curve's force there should be close to fraction * f_peak.
    curve = _tetralinear()
    f_peak = curve[:, 1].max()
    bb = fit_cbf_knots_backbone(
        curve, joint_force_fraction=0.90, residual_force_fraction=0.20).fitted_po
    d, f = curve[:, 0], curve[:, 1]
    assert np.interp(bb[2, 0], d, f) == pytest.approx(0.90 * f_peak, rel=0.05)
    assert np.interp(bb[3, 0], d, f) == pytest.approx(0.20 * f_peak, rel=0.05)


def test_recovers_a_noise_free_tetralinear_curve():
    bb = fit_cbf_knots_backbone(
        _tetralinear(), joint_force_fraction=0.90, residual_force_fraction=0.16).fitted_po
    assert bb[1, 0] == pytest.approx(20.0, abs=0.5)     # d_y
    assert bb[1, 1] == pytest.approx(1000.0, rel=0.02)  # f_y
    assert bb[2, 0] == pytest.approx(100.0, abs=1.0)    # d_f
    assert bb[2, 1] == pytest.approx(900.0, rel=0.02)   # f_f


def test_warns_when_the_residual_level_is_never_crossed():
    # _tetralinear() ends at 15 % of the peak: no 5 % crossing. The warning
    # gives the level the curve ends at, so the user can pick a higher one.
    match = r"Raise residual_force_fraction \(the curve ends at 0\.15"
    with pytest.warns(UserWarning, match=match):
        res = fit_cbf_knots_backbone(_tetralinear())
    assert len(res.fitted_po) == 4
    # With a level the curve does cross, there is no warning and a real plateau.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fit_cbf_knots_backbone(_tetralinear(), residual_force_fraction=0.20)
    assert len(res.fitted_po) == 5


def test_raises_when_the_joint_level_is_never_crossed():
    # Never falls to 80 % of the peak (ends at 90 %): no joint knot to snap to.
    curve = _tetralinear(f_r=900.0)
    for plateau in (True, False):
        with pytest.raises(ValueError, match="No descending crossing of joint_force_fraction"):
            fit_cbf_knots_backbone(curve, residual_plateau=plateau)


@pytest.mark.parametrize("jff,rff", [(0.8, 0.9), (0.5, 0.5), (0.0, 0.5), (1.0, 0.5)])
def test_rejects_invalid_fractions(jff, rff):
    with pytest.raises(ValueError, match="residual_force_fraction"):
        fit_cbf_knots_backbone(_tetralinear(),
                                 joint_force_fraction=jff,
                                 residual_force_fraction=rff)


def test_rejects_a_curve_with_no_descending_branch():
    d = np.linspace(0.0, 100.0, 50)
    rising = np.column_stack([d, d * 10.0])
    with pytest.raises(ValueError, match="No descending branch"):
        fit_cbf_knots_backbone(rising)


def test_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 5 points"):
        fit_cbf_knots_backbone(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]]))


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        fit_cbf_knots_backbone(np.zeros((10, 3)))


# ---------------------------------------------------------------------------
# The explicit-knot path
# ---------------------------------------------------------------------------


def test_explicit_knots_are_used_verbatim():
    bb = fit_cbf_knots_backbone(_tetralinear(), d_f=95.0, d_r=130.0).fitted_po
    assert bb[2, 0] == pytest.approx(95.0)
    assert bb[3, 0] == pytest.approx(130.0)


def test_explicit_knots_need_not_be_data_points():
    # 401 points over 0..200 -> spacing 0.5 mm. Ask for knots between samples.
    bb = fit_cbf_knots_backbone(_tetralinear(), d_f=95.137, d_r=130.271).fitted_po
    assert bb[2, 0] == pytest.approx(95.137)
    assert bb[3, 0] == pytest.approx(130.271)


def test_explicit_knots_reproduce_the_snapped_fit_exactly():
    """The two mechanisms agree when pointed at the same knots.

    This is the guard that the explicit path is the same fit, differently
    addressed -- not a different fit.
    """
    curve = _tetralinear()
    jff, rff = 0.90, 0.20
    by_fraction = fit_cbf_knots_backbone(
        curve, joint_force_fraction=jff, residual_force_fraction=rff).fitted_po

    # Resolve the fractions to knots the way the snapping path does.
    f = curve[:, 1]
    i_peak = int(np.argmax(f))
    f_peak = float(f[i_peak])
    i_f = _snap_descending_crossing(f, i_peak, jff * f_peak)
    i_r = _snap_descending_crossing(f, i_f, rff * f_peak)

    by_knots = fit_cbf_knots_backbone(
        curve, d_f=float(curve[i_f, 0]), d_r=float(curve[i_r, 0])).fitted_po
    np.testing.assert_allclose(by_knots, by_fraction, rtol=1e-9, atol=1e-9)


def test_explicit_knots_recover_known_breakpoints():
    bb = fit_cbf_knots_backbone(_tetralinear(), d_f=100.0, d_r=120.0).fitted_po
    assert bb[1, 0] == pytest.approx(20.0, abs=0.2)     # d_y
    assert bb[1, 1] == pytest.approx(1000.0, rel=0.01)  # f_y
    assert bb[2, 0] == pytest.approx(100.0)             # d_f, given
    assert bb[2, 1] == pytest.approx(900.0, rel=0.01)   # f_f, fitted
    assert bb[3, 0] == pytest.approx(120.0)             # d_r, given


def test_explicit_knots_ignore_the_force_fractions():
    # Fractions that would be rejected on the snapping path are simply unused.
    bb = fit_cbf_knots_backbone(
        _tetralinear(), joint_force_fraction=0.1, residual_force_fraction=0.9,
        d_f=100.0, d_r=120.0).fitted_po
    assert bb[2, 0] == pytest.approx(100.0)
    assert bb[3, 0] == pytest.approx(120.0)


def test_explicit_knots_are_monotone_in_candidate_resolution():
    """Densifying the curve cannot worsen the best achievable explicit-knot fit.

    This is the property the explicit path exists for. Inserting points along
    the segments leaves the piecewise-linear curve unchanged, so the same knots
    must give the same fit -- unlike the fraction path, where the snapping can
    resolve a fraction to a different point once the sampling changes.
    """
    coarse = _tetralinear(n=101)
    fine = _tetralinear(n=1601)
    a = fit_cbf_knots_backbone(coarse, d_f=100.0, d_r=120.0).fitted_po
    b = fit_cbf_knots_backbone(fine, d_f=100.0, d_r=120.0).fitted_po
    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("d_f,d_r", [
    (0.0, 120.0),       # d_f must be positive
    (-5.0, 120.0),
    (120.0, 100.0),     # d_f must precede d_r
    (100.0, 100.0),
    (100.0, 250.0),     # d_r beyond the end of the curve
])
def test_rejects_invalid_explicit_knots(d_f, d_r):
    with pytest.raises(ValueError, match="0 < d_f < d_r <= d_max"):
        fit_cbf_knots_backbone(_tetralinear(), d_f=d_f, d_r=d_r)


@pytest.mark.parametrize("kwargs", [{"d_f": 100.0}, {"d_r": 120.0}])
def test_rejects_one_knot_without_the_other(kwargs):
    with pytest.raises(ValueError, match="both d_f and d_r"):
        fit_cbf_knots_backbone(_tetralinear(), **kwargs)


def test_explicit_knots_work_on_the_example_curves():
    # Exercise the path on real data rather than only synthetic curves.
    paths = sorted(CURVES_DIR.glob("*.csv"))
    if not paths:
        pytest.skip(f"no example curves in {CURVES_DIR}")
    for p in paths:
        curve = np.loadtxt(p, delimiter=",")
        d = curve[:, 0]
        i_peak = int(np.argmax(curve[:, 1]))
        if i_peak >= len(d) - 3:
            continue
        # Two knots in the descending branch, at a third and two thirds of it.
        d_peak, d_max = float(d[i_peak]), float(d[-1])
        d_f = d_peak + (d_max - d_peak) / 3.0
        d_r = d_peak + 2.0 * (d_max - d_peak) / 3.0
        bb = fit_cbf_knots_backbone(curve, d_f=d_f, d_r=d_r, origin="correct_disp").fitted_po
        assert bb.shape == (5, 2)
        # The knot is given in input coordinates; fitted_po is shifted by d[0].
        assert bb[2, 0] == pytest.approx(d_f - d[0])
        assert bb[3, 0] == pytest.approx(d_r - d[0])
        assert np.all(np.isfinite(bb))


# ---------------------------------------------------------------------------
# The drop-to-zero mode (residual_plateau=False)
# ---------------------------------------------------------------------------


def _drop_curve(d_zero: float, d_max: float = 200.0, n: int = 401) -> np.ndarray:
    """Noise-free curve: elastic to (20, 1000), decline to (100, 900), drop to zero at d_zero."""
    xs = np.asarray([0.0, 20.0, 100.0, d_zero, max(d_zero, d_max) + 1.0])
    ys = np.asarray([0.0, 1000.0, 900.0, 0.0, 0.0])
    d = np.linspace(0.0, d_max, n)
    return np.column_stack([d, np.interp(d, xs, ys)])


def test_drop_mode_recovers_a_drop_that_ends_inside_the_data():
    res = fit_cbf_knots_backbone(_drop_curve(140.0), residual_plateau=False, d_f=100.0)
    assert res.mode == "trilinear-cbf-knots"
    assert res.converged
    # [0, 0], yield, joint, zero-force point, then [d_max, 0].
    expected = [[0.0, 0.0], [20.0, 1000.0], [100.0, 900.0], [140.0, 0.0], [200.0, 0.0]]
    np.testing.assert_allclose(res.fitted_po, expected, rtol=1e-3, atol=1e-6)


def test_drop_mode_recovers_a_drop_that_ends_beyond_the_data():
    # The line reaches zero at d = 250, past d_max = 200: no [d_max, 0] row.
    res = fit_cbf_knots_backbone(_drop_curve(250.0), residual_plateau=False, d_f=100.0)
    expected = [[0.0, 0.0], [20.0, 1000.0], [100.0, 900.0], [250.0, 0.0]]
    np.testing.assert_allclose(res.fitted_po, expected, rtol=1e-3, atol=1e-6)


def test_drop_mode_needs_no_residual_crossing():
    # _tetralinear() never falls below 15 % of the peak, so a 5 % residual
    # crossing does not exist; the drop mode doesn't look for one.
    res = fit_cbf_knots_backbone(_tetralinear(), residual_plateau=False)
    assert res.converged
    assert res.fitted_po[-1, 1] == 0.0
    assert (np.diff(res.fitted_po[:, 0]) > 0.0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"residual_force_fraction": 0.95},  # invalid for the plateau mode
        {"residual_force_fraction": -1.0},
        {"d_f": 100.0, "d_r": 50.0},  # d_r before d_f: ignored
        {"d_f": 100.0, "d_r": 1e9},  # d_r beyond the data: ignored
    ],
)
def test_drop_mode_ignores_d_r_and_the_residual_fraction(kwargs):
    curve = _drop_curve(140.0)
    reference = fit_cbf_knots_backbone(
        curve, residual_plateau=False, d_f=kwargs.get("d_f")
    )
    res = fit_cbf_knots_backbone(curve, residual_plateau=False, **kwargs)
    np.testing.assert_array_equal(res.fitted_po, reference.fitted_po)


def test_drop_mode_scores_only_the_data_range():
    # The drop of this curve reaches zero past d_max, so the fitted line extends
    # beyond the data. Only [d_y, d_max] may be scored: nothing past d_max,
    # where the data would otherwise be extended flat at f[-1].
    curve = _drop_curve(250.0)
    d, f = curve[:, 0], curve[:, 1]
    res = fit_cbf_knots_backbone(curve, residual_plateau=False)
    assert res.fitted_po[-1, 0] > d[-1]
    d_y = res.fitted_po[1, 0]
    assert res.objective == pytest.approx(
        _interval_area_error(d, f, res.fitted_po, d_y, d[-1]), rel=1e-12
    )
    # An independent dense integral over [d_y, d_max] gives the same score.
    x = np.linspace(d_y, d[-1], 200001)
    fit = np.interp(x, res.fitted_po[:, 0], res.fitted_po[:, 1])
    dense = np.trapezoid(np.abs(np.interp(x, d, f) - fit), x)
    assert res.objective == pytest.approx(dense, rel=1e-4)


def test_drop_mode_ignores_fake_points_appended_beyond_the_fit_range():
    # A noisy curve whose drop ends inside the data: points appended at zero
    # force beyond d_max lie on the fitted zero-force tail, so they add no
    # area and the knot forces and drop end are unchanged.
    rng = np.random.default_rng(3)
    curve = _drop_curve(140.0)
    curve[1:, 1] += rng.normal(0.0, 5.0, len(curve) - 1)
    extra = np.column_stack([np.linspace(201.0, 260.0, 60), np.zeros(60)])
    a = fit_cbf_knots_backbone(curve, residual_plateau=False, d_f=100.0).fitted_po
    b = fit_cbf_knots_backbone(
        np.vstack([curve, extra]), residual_plateau=False, d_f=100.0
    ).fitted_po
    np.testing.assert_allclose(b[:4], a[:4], rtol=1e-6)
    assert b[-1].tolist() == [260.0, 0.0]


def test_drop_mode_rejects_d_f_outside_the_data():
    with pytest.raises(ValueError, match="0 < d_f < d_max"):
        fit_cbf_knots_backbone(_drop_curve(140.0), residual_plateau=False, d_f=250.0)
