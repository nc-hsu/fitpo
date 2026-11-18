"""Tests for the shared interface: FitResult, res.mode, area_error, keyword-only args."""

import numpy as np
import pytest

import fitpo
from fitpo import FitResult, area_error


def test_public_interface():
    # The top-level package is the only public interface.
    assert set(fitpo.__all__) == {
        "FitResult",
        "__version__",
        "area_error",
        "elastic_slope",
        "fit_bilinear_backbone",
        "fit_cbf_cusum_backbone",
        "fit_cbf_knots_backbone",
        "fit_deluca_backbone",
        "fit_envelope",
    }


def test_deluca_modes(smooth_quadrilinear, smooth_trilinear):
    # Auto reports the shape it chose, not "auto".
    assert fitpo.fit_deluca_backbone(smooth_quadrilinear.curve).mode == "quadrilinear-deluca"
    assert fitpo.fit_deluca_backbone(smooth_trilinear.curve).mode == "trilinear-deluca"
    for mode in ("trilinear-deluca", "quadrilinear-deluca"):
        res = fitpo.fit_deluca_backbone(smooth_quadrilinear.curve, mode=mode)
        assert isinstance(res, FitResult)
        assert res.mode == mode


def test_bilinear_modes(bilinear_source):
    for mode in ("bilinear-ultimate", "bilinear-pctloss"):
        assert fitpo.fit_bilinear_backbone(bilinear_source.curve, mode=mode).mode == mode
    # The default is bilinear-ultimate.
    assert fitpo.fit_bilinear_backbone(bilinear_source.curve).mode == "bilinear-ultimate"


def _cbf_curve():
    xs = [0.0, 20.0, 100.0, 120.0, 200.0]
    ys = [0.0, 1000.0, 900.0, 150.0, 150.0]
    d = np.linspace(0.0, 200.0, 401)
    return np.column_stack([d, np.interp(d, xs, ys)])


def test_cbf_knots_modes():
    curve = _cbf_curve()
    assert fitpo.fit_cbf_knots_backbone(curve).mode == "quadrilinear-cbf-knots"
    res = fitpo.fit_cbf_knots_backbone(curve, residual_plateau=False)
    assert res.mode == "trilinear-cbf-knots"


def test_cbf_cusum_mode():
    # A noisy CBF-like curve with a sharp fracture drop, which CUSUM detects.
    xs = [0.0, 10.0, 80.0, 82.0, 120.0]
    ys = [0.0, 500.0, 350.0, 100.0, 80.0]
    d = np.linspace(0.0, 120.0, 801)
    f = np.interp(d, xs, ys) + np.random.default_rng(7).normal(0.0, 3.5, d.size)
    f[0] = 0.0  # the unloaded first point is exact
    res = fitpo.fit_cbf_cusum_backbone(np.column_stack([d, f]))
    assert res.mode == "quadrilinear-cbf-cusum"


@pytest.mark.parametrize(
    "func",
    [
        fitpo.fit_deluca_backbone,
        fitpo.fit_bilinear_backbone,
    ],
)
def test_invalid_mode_raises(func, smooth_quadrilinear):
    with pytest.raises(ValueError, match="mode must be one of"):
        func(smooth_quadrilinear.curve, mode="trilinear")


def test_mode_is_keyword_only(smooth_quadrilinear):
    with pytest.raises(TypeError):
        fitpo.fit_deluca_backbone(smooth_quadrilinear.curve, "trilinear-deluca")  # type: ignore[misc]


def test_objective_and_converged(smooth_quadrilinear):
    res = fitpo.fit_deluca_backbone(smooth_quadrilinear.curve)
    assert res.converged is True
    assert res.objective > 0.0
    # objective covers [d_y, d_max] only, so it can't exceed the whole-curve error.
    assert res.objective <= area_error(smooth_quadrilinear.curve, res.fitted_po) + 1e-9


def test_area_error_is_zero_for_an_exact_fit():
    bp = np.asarray([[0.0, 0.0], [1.0, 100.0], [3.0, 150.0], [10.0, 0.0]])
    d = np.linspace(0.0, 10.0, 51)  # step 0.2 hits every breakpoint
    curve = np.column_stack([d, np.interp(d, bp[:, 0], bp[:, 1])])
    assert area_error(curve, bp) == pytest.approx(0.0, abs=1e-9)


def test_area_error_known_value_and_normalised():
    # Curve F = 10 on [0, 4]. The fit rises from 0 to 10 over [0, 2] and then
    # matches, so the error is a triangle of area 0.5 * 2 * 10 = 10.
    curve = np.asarray([[0.0, 10.0], [4.0, 10.0]])
    bp = np.asarray([[0.0, 0.0], [2.0, 10.0], [4.0, 10.0]])
    # origin="as-is": the curve's non-zero first force is the point here.
    assert area_error(curve, bp, origin="as-is") == pytest.approx(10.0)
    # Area under the curve is 40.
    assert area_error(curve, bp, normalised=True, origin="as-is") == pytest.approx(0.25)


def test_area_error_checks_fitted_shape():
    with pytest.raises(ValueError, match="fitted_po must be shape"):
        area_error(np.zeros((5, 2)), np.zeros((3, 3)))
