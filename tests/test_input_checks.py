"""Tests for the input checks, the origin correction and the elastic_slope crossing rule."""

import numpy as np
import pytest

import fitpo

# These tests are about input handling, not fit quality. The noise-free curves
# below peak exactly at yield, so De Luca has no hardening line and warns, and
# they never fall to 5 % of the peak, so the knots fit warns that it has no
# residual plateau. Both warnings are tested elsewhere.
pytestmark = [
    pytest.mark.filterwarnings("ignore:trilinear/quadrilinear need a hardening branch"),
    pytest.mark.filterwarnings("ignore:.*the force never falls to residual_force_fraction"),
]


def _cbf_curve(d0: float = 0.0, f0: float = 0.0) -> np.ndarray:
    """Noise-free CBF-like curve, optionally offset so it starts at (d0, f0)."""
    xs = [0.0, 20.0, 100.0, 120.0, 200.0]
    ys = [0.0, 1000.0, 900.0, 150.0, 150.0]
    d = np.linspace(0.0, 200.0, 401)
    return np.column_stack([d + d0, np.interp(d, xs, ys) + f0])


def _cusum_curve(d0: float = 0.0, f0: float = 0.0) -> np.ndarray:
    """Noisy curve with a sharp fracture drop that CUSUM detects, starting at (d0, f0)."""
    xs = [0.0, 10.0, 80.0, 82.0, 120.0]
    ys = [0.0, 500.0, 350.0, 100.0, 80.0]
    d = np.linspace(0.0, 120.0, 801)
    f = np.interp(d, xs, ys) + np.random.default_rng(7).normal(0.0, 3.5, d.size)
    f[0] = 0.0
    return np.column_stack([d + d0, f + f0])


# Every backbone function, called with its default mode, on a curve it can fit.
BACKBONES = [
    pytest.param(fitpo.fit_deluca_backbone, _cbf_curve, id="deluca"),
    pytest.param(fitpo.fit_bilinear_backbone, _cbf_curve, id="bilinear"),
    pytest.param(fitpo.fit_cbf_cusum_backbone, _cusum_curve, id="cbf_cusum"),
    pytest.param(fitpo.fit_cbf_knots_backbone, _cbf_curve, id="cbf_knots"),
]


# ---------------------------------------------------------------------------
# NaN / inf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("func,make", BACKBONES)
def test_backbones_reject_non_finite(func, make, bad):
    curve = make()
    curve[50, 1] = bad
    with pytest.raises(ValueError, match="NaN or inf"):
        func(curve)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_area_error_rejects_non_finite(bad):
    curve = _cbf_curve()
    bp = np.asarray([[0.0, 0.0], [20.0, 1000.0], [200.0, 150.0]])
    bad_curve = curve.copy()
    bad_curve[10, 0] = bad
    with pytest.raises(ValueError, match="NaN or inf"):
        fitpo.area_error(bad_curve, bp)
    bad_bp = bp.copy()
    bad_bp[1, 1] = bad
    with pytest.raises(ValueError, match="NaN or inf"):
        fitpo.area_error(curve, bad_bp)


def test_envelope_rejects_non_finite(fracture_cyclic):
    curve = fracture_cyclic.curve.copy()
    curve[20, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        fitpo.fit_envelope(curve)


def test_elastic_slope_rejects_non_finite():
    d = np.linspace(0.0, 10.0, 11)
    f = d * 10.0
    f[3] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        fitpo.elastic_slope(d, f, 100.0)
    with pytest.raises(ValueError, match="NaN or inf"):
        fitpo.elastic_slope(d, d * 10.0, np.inf)


# ---------------------------------------------------------------------------
# Decreasing displacement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func,make", BACKBONES)
def test_backbones_reject_decreasing_displacement(func, make):
    curve = make()
    curve[60, 0] = curve[58, 0]  # d[60] < d[59]
    with pytest.raises(ValueError, match="never decrease"):
        func(curve)


@pytest.mark.parametrize("func,make", BACKBONES)
def test_backbones_allow_repeated_displacement(func, make):
    # A repeated displacement (two steps at the same d) is valid data.
    curve = make()
    curve = np.insert(curve, 61, curve[60], axis=0)
    func(curve)


def test_area_error_rejects_decreasing_displacement():
    curve = _cbf_curve()
    bp = np.asarray([[0.0, 0.0], [20.0, 1000.0], [200.0, 150.0]])
    bad = curve.copy()
    bad[60, 0] = bad[58, 0]
    with pytest.raises(ValueError, match="never decrease"):
        fitpo.area_error(bad, bp)
    with pytest.raises(ValueError, match="never decrease"):
        fitpo.area_error(curve, bp[[0, 2, 1]])


def test_envelope_accepts_reversing_displacement(fracture_cyclic):
    # Cyclic data reverses by design; fit_envelope must not reject it.
    pos, neg = fitpo.fit_envelope(fracture_cyclic.curve)
    assert len(pos) > 1 and len(neg) > 1


# ---------------------------------------------------------------------------
# origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func,make", BACKBONES)
@pytest.mark.parametrize("origin", ["strict", "correct_disp", "correct_dispforc", "as-is"])
def test_every_origin_mode_agrees_on_a_curve_at_the_origin(func, make, origin):
    # A curve that starts at (0, 0) needs no correction, so all modes match.
    curve = make()
    ref = func(curve, origin="as-is")
    np.testing.assert_array_equal(func(curve, origin=origin).fitted_po, ref.fitted_po)


@pytest.mark.parametrize("func,make", BACKBONES)
def test_strict_is_the_default_and_reports_the_offset(func, make):
    # The message gives the first point and how far out it is, in percent.
    curve = make(d0=0.5)
    with pytest.raises(ValueError, match=r"must start at \(0, 0\).*\(0\.5, 0\)") as err:
        func(curve)
    assert "%" in str(err.value) and "correct_disp" in str(err.value)
    with pytest.raises(ValueError, match="must start at"):
        func(curve, origin="strict")


@pytest.mark.parametrize("func,make", BACKBONES)
def test_strict_rejects_a_force_offset_alone(func, make):
    curve = make()
    curve[0, 1] = 1.0  # displacement is at 0, force is not
    with pytest.raises(ValueError, match="must start at"):
        func(curve)


@pytest.mark.parametrize("func,make", BACKBONES)
def test_correct_disp_shifts_displacement_only(func, make):
    # A pure displacement offset is removed exactly.
    at_origin = func(make())
    shifted = func(make(d0=0.5), origin="correct_disp")
    np.testing.assert_allclose(shifted.fitted_po, at_origin.fitted_po, rtol=1e-6, atol=1e-6)
    # A force offset is kept, so the fit differs from the one at the origin.
    raised = func(make(d0=0.5, f0=3.0), origin="correct_disp")
    assert raised.fitted_po[0].tolist() == [0.0, 0.0]
    same_shape = raised.fitted_po.shape == at_origin.fitted_po.shape
    assert not same_shape or not np.allclose(raised.fitted_po, at_origin.fitted_po)


@pytest.mark.parametrize("func,make", BACKBONES)
def test_correct_dispforc_removes_the_whole_first_point(func, make):
    at_origin = func(make())
    shifted = func(make(d0=0.5, f0=3.0), origin="correct_dispforc")
    np.testing.assert_allclose(shifted.fitted_po, at_origin.fitted_po, rtol=1e-6, atol=1e-6)
    assert shifted.fitted_po[0].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("func,make", BACKBONES)
def test_as_is_counts_the_offset_as_elastic(func, make):
    # "as-is" anchors the elastic line at the true origin, so a displacement
    # offset counts as elastic deformation and the stiffness drops.
    offset = make(d0=2.0)
    shifted = func(offset, origin="correct_disp").fitted_po
    as_given = func(offset, origin="as-is").fitted_po
    assert as_given[1, 1] / as_given[1, 0] < shifted[1, 1] / shifted[1, 0]


@pytest.mark.parametrize("func,make", BACKBONES)
def test_unknown_origin_raises(func, make):
    with pytest.raises(ValueError, match="origin must be one of"):
        func(make(), origin="shift")


@pytest.mark.parametrize("origin", ["correct_disp", "correct_dispforc"])
def test_knots_are_shifted_with_the_curve(origin):
    # Explicit knots are given in input coordinates and move with the shift.
    d0 = 5.0
    curve = _cbf_curve(d0=d0)
    res = fitpo.fit_cbf_knots_backbone(curve, d_f=100.0 + d0, d_r=120.0 + d0, origin=origin)
    assert res.fitted_po[2, 0] == pytest.approx(100.0)
    assert res.fitted_po[3, 0] == pytest.approx(120.0)
    ref = fitpo.fit_cbf_knots_backbone(_cbf_curve(), d_f=100.0, d_r=120.0)
    np.testing.assert_allclose(res.fitted_po, ref.fitted_po, rtol=1e-6, atol=1e-6)


def test_knots_are_used_as_given_with_as_is():
    d0 = 5.0
    raw = fitpo.fit_cbf_knots_backbone(
        _cbf_curve(d0=d0), d_f=100.0 + d0, d_r=120.0 + d0, origin="as-is"
    )
    assert raw.fitted_po[2, 0] == pytest.approx(100.0 + d0)


def test_area_error_origin():
    curve = _cbf_curve(d0=0.5, f0=3.0)
    res = fitpo.fit_cbf_knots_backbone(curve, origin="correct_dispforc")
    shifted = curve - curve[0]
    # With the same setting as the fit, area_error compares like with like.
    assert fitpo.area_error(curve, res.fitted_po, origin="correct_dispforc") == pytest.approx(
        fitpo.area_error(shifted, res.fitted_po)
    )
    with pytest.raises(ValueError, match="must start at"):
        fitpo.area_error(curve, res.fitted_po)
    with pytest.raises(ValueError, match="origin must be one of"):
        # Deliberately not an OriginMode value: the runtime check is under test.
        fitpo.area_error(shifted, res.fitted_po, origin="shift")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# elastic_slope: first crossing
# ---------------------------------------------------------------------------


def test_elastic_slope_uses_the_first_crossing():
    # The force passes 10 % of peak between d = 3 and d = 4, dips back below it,
    # then rises to the peak. np.interp with force as x-axis is undefined here
    # and picks a later crossing on this data; the first-crossing rule gives
    # d_ref = 3 + (10 - 6) / (15 - 6) = 3.444..., so k = 10 / d_ref.
    d = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    f = np.asarray([0.0, 3.0, 5.0, 6.0, 15.0, 5.0, 36.0, 100.0])
    k_expected = 10.0 / (3.0 + 4.0 / 9.0)
    assert fitpo.elastic_slope(d, f, 100.0) == pytest.approx(k_expected)
    assert 10.0 / np.interp(10.0, f, d) != pytest.approx(k_expected)


def test_elastic_slope_interpolates_between_samples():
    d = np.asarray([0.0, 1.0, 2.0, 3.0])
    f = np.asarray([0.0, 4.0, 16.0, 100.0])
    # 10 % of 100 is 10, reached between d = 1 (4) and d = 2 (16): d_ref = 1.5.
    assert fitpo.elastic_slope(d, f, 100.0) == pytest.approx(10.0 / 1.5)


def test_elastic_slope_rejects_a_crossing_before_the_data():
    # The first sample (2, 30) is already above 10 % of the 100 peak.
    d = np.asarray([2.0, 3.0, 4.0])
    f = np.asarray([30.0, 60.0, 100.0])
    with pytest.raises(ValueError, match="lies before the data"):
        fitpo.elastic_slope(d, f, 100.0)


@pytest.mark.parametrize("func,make", BACKBONES)
def test_backbones_reject_a_crossing_before_the_data(func, make):
    # Used as-is, a curve starting at 20 % of peak force has no elastic
    # reference point inside the data; with the force offset removed it fits.
    curve = make()
    curve[:, 1] += 0.2 * curve[:, 1].max()
    with pytest.raises(ValueError, match="lies before the data"):
        func(curve, origin="as-is")
    func(curve, origin="correct_dispforc")


def test_elastic_slope_ref_fraction():
    d = np.linspace(0.0, 10.0, 101)
    f = 10.0 * d
    assert fitpo.elastic_slope(d, f, 100.0, ref_fraction=0.3) == pytest.approx(10.0)


def test_backbones_handle_non_monotone_force_below_ref():
    # A dip below 10 % of peak on the rising branch must not break the fits.
    curve = _cbf_curve()
    curve[3, 1] = curve[5, 1] * 1.2  # bump above, then dip back
    curve[4, 1] = curve[2, 1]
    # 10 % of the 1000 peak is first crossed between d = 1.0 (f = 50) and
    # d = 1.5 (f = 150), at d = 1.25, so k_e = 100 / 1.25 = 80. The yield point
    # lies on the elastic line, so f_y / d_y must equal it.
    for func in (
        fitpo.fit_deluca_backbone,
        fitpo.fit_bilinear_backbone,
        fitpo.fit_cbf_knots_backbone,
    ):
        d_y, f_y = func(curve).fitted_po[1]
        assert f_y / d_y == pytest.approx(80.0)
