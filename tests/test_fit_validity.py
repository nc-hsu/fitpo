"""Regression tests: every fit returns a valid shape, and the De Luca fits are near-optimal.

The validity rules are the ones every backbone result must meet: finite
values, the origin as the first row, a positive yield displacement, and
displacement strictly increasing from row to row. The near-optimality test
compares the De Luca ``objective`` with a dense, independent grid search.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

import fitpo
from fitpo._common import _finish, _interval_area_error

CURVES = sorted((Path(__file__).resolve().parent / "data").glob("*.csv"))
NO_HARDENING = "trilinear/quadrilinear need a hardening branch"
NO_RESIDUAL_CROSSING = ".*the force never falls to residual_force_fraction"


def _all_fits(curve: np.ndarray, origin: str) -> dict[str, fitpo.FitResult]:
    """Every backbone function in every mode, with the given origin setting."""
    fits = {}
    for mode in ("auto", "bilinear-deluca", "trilinear-deluca", "quadrilinear-deluca"):
        fits[mode] = fitpo.fit_deluca_backbone(curve, mode=mode, origin=origin)
    for mode in ("bilinear-ultimate", "bilinear-pctloss"):
        fits[mode] = fitpo.fit_bilinear_backbone(curve, mode=mode, origin=origin)
    fits["cbf-cusum"] = fitpo.fit_cbf_cusum_backbone(curve, origin=origin)
    for plateau in (True, False):
        res = fitpo.fit_cbf_knots_backbone(curve, residual_plateau=plateau, origin=origin)
        fits[res.mode] = res
    return fits


def _assert_valid(bp: np.ndarray, label: str) -> None:
    assert np.isfinite(bp).all(), label
    assert bp[0].tolist() == [0.0, 0.0], label
    assert bp[1, 0] > 0.0, label
    assert (np.diff(bp[:, 0]) > 0.0).all(), f"{label}: {bp[:, 0]}"


# ---------------------------------------------------------------------------
# Valid geometry on the nine example curves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("origin", ["correct_disp", "as-is"])
@pytest.mark.parametrize("path", CURVES, ids=[p.stem for p in CURVES])
def test_every_mode_gives_valid_geometry(path, origin):
    curve = np.loadtxt(path, delimiter=",")
    with warnings.catch_warnings():
        # po_curve_61 as-is has no hardening line; that case is tested below.
        warnings.filterwarnings("ignore", NO_HARDENING)
        # Most curves never fall to 5 % of the peak, so the knots fit warns
        # that it has no residual plateau; tested in test_tetralinear_fitting.
        warnings.filterwarnings("ignore", NO_RESIDUAL_CROSSING)
        fits = _all_fits(curve, origin)
    d_max = curve[-1, 0] - (curve[0, 0] if origin == "correct_disp" else 0.0)
    for mode, res in fits.items():
        _assert_valid(res.fitted_po, f"{path.stem} {origin} {mode}")
        assert res.objective >= 0.0
        # Every fit covers at least [0, d_max], except bilinear-pctloss, which
        # by design ends where the force has dropped to the set fraction.
        if mode != "bilinear-pctloss":
            assert res.fitted_po[-1, 0] >= d_max * (1 - 1e-12), f"{path.stem} {mode}"
        # The De Luca fits never go beyond the data or above the peak.
        if mode.endswith("deluca") or mode == "auto":
            assert res.fitted_po[-1, 0] == pytest.approx(curve[-1, 0] - (
                curve[0, 0] if origin == "correct_disp" else 0.0))
            assert res.fitted_po[:, 1].max() <= curve[:, 1].max() * (1 + 1e-12)


# ---------------------------------------------------------------------------
# No hardening line: po_curve_61 used as-is
# ---------------------------------------------------------------------------


def _po61() -> np.ndarray:
    return np.loadtxt(Path(__file__).resolve().parent / "data" / "po_curve_61.csv", delimiter=",")


@pytest.mark.parametrize(
    "mode,rows,fitted",
    [
        ("trilinear-deluca", 3, "trilinear-deluca"),
        ("quadrilinear-deluca", 4, "quadrilinear-deluca"),
        ("auto", None, None),
    ],
)
def test_no_hardening_line_warns_and_yields_at_peak_force(mode, rows, fitted):
    curve = _po61()
    d, f = curve[:, 0], curve[:, 1]
    i_peak = int(np.argmax(f))
    f_max = float(f[i_peak])
    k_e = fitpo.elastic_slope(d[: i_peak + 1], f[: i_peak + 1], f_max)
    assert f_max / d[i_peak] >= k_e  # the case under test

    with pytest.warns(UserWarning, match=NO_HARDENING):
        res = fitpo.fit_deluca_backbone(curve, mode=mode, origin="as-is")
    # Yield strength = maximum strength, stiffness kept at k_e, no peak row.
    np.testing.assert_allclose(res.fitted_po[1], [f_max / k_e, f_max])
    assert res.fitted_po[2, 1] < f_max
    assert res.mode in ("trilinear-deluca", "quadrilinear-deluca")
    if rows is not None:
        assert len(res.fitted_po) == rows
        assert res.mode == fitted
    _assert_valid(res.fitted_po, mode)


def test_no_hardening_raises_if_yield_is_beyond_the_data():
    # The secant to the last (peak) point is stiffer than k_e, and F_max / k_e
    # lies beyond d_max: no valid yield point exists.
    d = np.asarray([0.0, 1.0, 2.0, 3.0, 3.5, 3.6])
    f = np.asarray([0.0, 10.0, 40.0, 100.0, 90.0, 80.0])
    with pytest.raises(ValueError, match="beyond the last data point"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", NO_HARDENING)
        fitpo.fit_deluca_backbone(np.column_stack([d, f]), mode="trilinear-deluca")


# ---------------------------------------------------------------------------
# bilinear-deluca
# ---------------------------------------------------------------------------


def test_bilinear_deluca_is_elastic_perfectly_plastic(smooth_quadrilinear):
    curve = smooth_quadrilinear.curve
    res = fitpo.fit_deluca_backbone(curve, mode="bilinear-deluca")
    d, f = curve[:, 0], curve[:, 1]
    i_peak = int(np.argmax(f))
    f_max = float(f[i_peak])
    k_e = fitpo.elastic_slope(d[: i_peak + 1], f[: i_peak + 1], f_max)
    np.testing.assert_allclose(
        res.fitted_po, [[0.0, 0.0], [f_max / k_e, f_max], [d[-1], f_max]]
    )
    assert res.mode == "bilinear-deluca"
    assert res.converged is True
    # Nothing is optimised; objective is the area error over [0, d_max].
    assert res.objective == pytest.approx(fitpo.area_error(curve, res.fitted_po))


def test_bilinear_deluca_works_with_the_peak_at_the_end(peak_at_end):
    res = fitpo.fit_deluca_backbone(peak_at_end.curve, mode="bilinear-deluca")
    _assert_valid(res.fitted_po, "peak at end")


# ---------------------------------------------------------------------------
# Result validation and the convergence warning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bp,match",
    [
        ([[0.0, 0.0], [-1.0, -5.0], [3.0, 1.0]], "yield displacement <= 0"),
        ([[0.0, 0.0], [2.0, 5.0], [1.0, 4.0]], "doesn't increase"),
        ([[0.0, 0.0], [1.0, 5.0], [1.0, 4.0]], "doesn't increase"),
        ([[0.0, 0.0], [1.0, np.nan], [2.0, 4.0]], "NaN or inf"),
    ],
)
def test_invalid_shapes_are_never_returned(bp, match):
    with pytest.raises(ValueError, match=match):
        _finish(np.asarray(bp), "test-mode", 1.0, True, "some_fit")


def test_exact_duplicate_rows_are_dropped():
    bp = np.asarray([[0.0, 0.0], [1.0, 5.0], [2.0, 4.0], [2.0, 4.0]])
    res = _finish(bp, "test-mode", 1.0, True, "some_fit")
    assert len(res.fitted_po) == 3


def test_non_convergence_warns_with_function_and_mode(monkeypatch, smooth_trilinear):
    import fitpo._deluca as deluca

    real = deluca.minimize_scalar

    def failing(*args, **kwargs):
        res = real(*args, **kwargs)
        res.success = False
        return res

    monkeypatch.setattr(deluca, "minimize_scalar", failing)
    with pytest.warns(UserWarning, match=r"fit_deluca_backbone \(mode 'trilinear-deluca'\)"):
        res = fitpo.fit_deluca_backbone(smooth_trilinear.curve, mode="trilinear-deluca")
    assert res.converged is False


# ---------------------------------------------------------------------------
# Near-optimality against a brute-force search
# ---------------------------------------------------------------------------


def _brute_force(curve: np.ndarray, plateau: bool) -> float:
    """Best De Luca objective over a dense grid of valid parameter values.

    The objective over [d_y, d_max] splits into the hardening part over
    [d_y, d_peak] (depends only on k_h) and the falling part over
    [d_peak, d_max] (depends only on the softening parameters), so each part
    is minimised on its own grid.
    """
    d, f = curve[:, 0], curve[:, 1]
    i_peak = int(np.argmax(f))
    d_peak, f_peak, d_max = float(d[i_peak]), float(f[i_peak]), float(d[-1])
    k_e = fitpo.elastic_slope(d[: i_peak + 1], f[: i_peak + 1], f_peak)

    hard = np.inf
    for k_h in np.linspace(0.0, f_peak / d_peak, 1501)[:-1]:
        d_y = (f_peak - k_h * d_peak) / (k_e - k_h)
        bp = np.asarray([[d_y, k_e * d_y], [d_peak, f_peak]])
        hard = min(hard, _interval_area_error(d, f, bp, d_y, d_peak))

    # Falling branch: the data on [d_peak, d_max] (both ends are data points).
    x = d[i_peak:]
    y = f[i_peak:]
    if not plateau:
        k_s = np.linspace(-10.0 * f_peak / (d_max - d_peak), 0.0, 20001)
        fit = f_peak + np.outer(k_s, x - d_peak)
        return hard + float(np.trapezoid(np.abs(y - fit), x, axis=1).min())

    best = np.inf
    f_p = np.linspace(0.0, f_peak, 301)[1:-1]
    for d_b in np.linspace(d_peak, d_max, 301)[1:-1]:
        k = int(np.searchsorted(x, d_b))
        g = np.insert(x, k, d_b)
        gy = np.insert(y, k, np.interp(d_b, d, f))
        t = np.minimum((g - d_peak) / (d_b - d_peak), 1.0)
        fit = f_peak * (1.0 - t) + np.outer(f_p, t)
        best = min(best, float(np.trapezoid(np.abs(gy - fit), g, axis=1).min()))
    return hard + best


@pytest.mark.parametrize(
    "name", ["smooth_quadrilinear", "smooth_trilinear", "bilinear_source", "staircase"]
)
@pytest.mark.parametrize("mode", ["trilinear-deluca", "quadrilinear-deluca"])
def test_objective_within_one_percent_of_brute_force(name, mode, request):
    curve = request.getfixturevalue(name).curve
    res = fitpo.fit_deluca_backbone(curve, mode=mode)
    brute = _brute_force(curve, plateau=mode == "quadrilinear-deluca")
    assert res.objective <= 1.01 * brute, f"{res.objective:.6g} vs brute {brute:.6g}"
