"""Tests for fit_tetralinear_backbone, including the explicit-knot path.

The function had no coverage before the ``d_f``/``d_r`` arguments were added,
so this covers both mechanisms: the original force-fraction snapping (which must
keep behaving exactly as it did) and the explicit knot displacements.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fitpo import fit_tetralinear_backbone
from fitpo.cbf_fitting import _snap_descending_crossing

CURVES_DIR = Path(__file__).resolve().parent.parent / "example_po_curves"


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
# The force-fraction path (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_returns_five_ordered_breakpoints():
    bb = fit_tetralinear_backbone(_tetralinear())
    assert bb.shape == (5, 2)
    assert bb[0].tolist() == [0.0, 0.0]
    # Displacements are non-decreasing, and the first three strictly increase.
    assert bb[0, 0] < bb[1, 0] < bb[2, 0] < bb[3, 0] <= bb[4, 0]


def test_knots_land_on_the_requested_force_fractions():
    # d_f and d_r are snapped to the descending crossings of the fractions,
    # so the curve's force there should be close to fraction * f_peak.
    curve = _tetralinear()
    f_peak = curve[:, 1].max()
    bb = fit_tetralinear_backbone(
        curve, joint_force_fraction=0.90, residual_force_fraction=0.20)
    d, f = curve[:, 0], curve[:, 1]
    assert np.interp(bb[2, 0], d, f) == pytest.approx(0.90 * f_peak, rel=0.05)
    assert np.interp(bb[3, 0], d, f) == pytest.approx(0.20 * f_peak, rel=0.05)


def test_recovers_a_noise_free_tetralinear_curve():
    bb = fit_tetralinear_backbone(
        _tetralinear(), joint_force_fraction=0.90, residual_force_fraction=0.16)
    assert bb[1, 0] == pytest.approx(20.0, abs=0.5)     # d_y
    assert bb[1, 1] == pytest.approx(1000.0, rel=0.02)  # f_y
    assert bb[2, 0] == pytest.approx(100.0, abs=1.0)    # d_f
    assert bb[2, 1] == pytest.approx(900.0, rel=0.02)   # f_f


def test_collapse_residual_drops_to_zero_force():
    bb = fit_tetralinear_backbone(_tetralinear(), collapse_residual=True)
    # Trilinear shape padded to 5x2 by repeating the endpoint at f = 0.
    assert bb[3, 1] == pytest.approx(0.0)
    assert bb[4].tolist() == bb[3].tolist()


@pytest.mark.parametrize("jff,rff", [(0.8, 0.9), (0.5, 0.5), (0.0, 0.5), (1.0, 0.5)])
def test_rejects_invalid_fractions(jff, rff):
    with pytest.raises(ValueError, match="residual_force_fraction"):
        fit_tetralinear_backbone(_tetralinear(),
                                 joint_force_fraction=jff,
                                 residual_force_fraction=rff)


def test_rejects_a_curve_with_no_descending_branch():
    d = np.linspace(0.0, 100.0, 50)
    rising = np.column_stack([d, d * 10.0])
    with pytest.raises(ValueError, match="No descending branch"):
        fit_tetralinear_backbone(rising)


def test_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 5 points"):
        fit_tetralinear_backbone(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]]))


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        fit_tetralinear_backbone(np.zeros((10, 3)))


# ---------------------------------------------------------------------------
# The explicit-knot path
# ---------------------------------------------------------------------------


def test_explicit_knots_are_used_verbatim():
    bb = fit_tetralinear_backbone(_tetralinear(), d_f=95.0, d_r=130.0)
    assert bb[2, 0] == pytest.approx(95.0)
    assert bb[3, 0] == pytest.approx(130.0)


def test_explicit_knots_need_not_be_data_points():
    # 401 points over 0..200 -> spacing 0.5 mm. Ask for knots between samples.
    bb = fit_tetralinear_backbone(_tetralinear(), d_f=95.137, d_r=130.271)
    assert bb[2, 0] == pytest.approx(95.137)
    assert bb[3, 0] == pytest.approx(130.271)


def test_explicit_knots_reproduce_the_snapped_fit_exactly():
    """The two mechanisms agree when pointed at the same knots.

    This is the guard that the explicit path is the same fit, differently
    addressed -- not a different fit.
    """
    curve = _tetralinear()
    jff, rff = 0.90, 0.20
    by_fraction = fit_tetralinear_backbone(
        curve, joint_force_fraction=jff, residual_force_fraction=rff)

    # Resolve the fractions to knots the way the snapping path does.
    f = curve[:, 1]
    i_peak = int(np.argmax(f))
    f_peak = float(f[i_peak])
    i_f = _snap_descending_crossing(f, i_peak, jff * f_peak)
    i_r = _snap_descending_crossing(f, i_f, rff * f_peak)

    by_knots = fit_tetralinear_backbone(
        curve, d_f=float(curve[i_f, 0]), d_r=float(curve[i_r, 0]))
    np.testing.assert_allclose(by_knots, by_fraction, rtol=1e-9, atol=1e-9)


def test_explicit_knots_recover_known_breakpoints():
    bb = fit_tetralinear_backbone(_tetralinear(), d_f=100.0, d_r=120.0)
    assert bb[1, 0] == pytest.approx(20.0, abs=0.2)     # d_y
    assert bb[1, 1] == pytest.approx(1000.0, rel=0.01)  # f_y
    assert bb[2, 0] == pytest.approx(100.0)             # d_f, given
    assert bb[2, 1] == pytest.approx(900.0, rel=0.01)   # f_f, fitted
    assert bb[3, 0] == pytest.approx(120.0)             # d_r, given


def test_explicit_knots_ignore_the_force_fractions():
    # Fractions that would be rejected on the snapping path are simply unused.
    bb = fit_tetralinear_backbone(
        _tetralinear(), joint_force_fraction=0.1, residual_force_fraction=0.9,
        d_f=100.0, d_r=120.0)
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
    a = fit_tetralinear_backbone(coarse, d_f=100.0, d_r=120.0)
    b = fit_tetralinear_backbone(fine, d_f=100.0, d_r=120.0)
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
        fit_tetralinear_backbone(_tetralinear(), d_f=d_f, d_r=d_r)


@pytest.mark.parametrize("kwargs", [{"d_f": 100.0}, {"d_r": 120.0}])
def test_rejects_one_knot_without_the_other(kwargs):
    with pytest.raises(ValueError, match="both d_f and d_r"):
        fit_tetralinear_backbone(_tetralinear(), **kwargs)


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
        bb = fit_tetralinear_backbone(curve, d_f=d_f, d_r=d_r)
        assert bb.shape == (5, 2)
        assert bb[2, 0] == pytest.approx(d_f)
        assert bb[3, 0] == pytest.approx(d_r)
        assert np.all(np.isfinite(bb))
