"""Tests for fit_piecewise_backbone."""

from __future__ import annotations

import numpy as np
import pytest

from fitpo import fit_piecewise_backbone


def _close(actual: float, expected: float, rel: float) -> bool:
    return abs(actual - expected) <= rel * max(abs(expected), 1e-12)


def test_auto_detects_quadrilinear(smooth_quadrilinear):
    fit = fit_piecewise_backbone(smooth_quadrilinear.curve)
    assert fit.shape == (5, 2)
    expected = smooth_quadrilinear.expected_breaks
    assert expected is not None
    for (d_a, f_a), (d_e, f_e) in zip(fit, expected, strict=True):
        assert _close(float(d_a), float(d_e), rel=0.15)
        assert _close(float(f_a), float(f_e), rel=0.15)


def test_auto_detects_trilinear(smooth_trilinear):
    fit = fit_piecewise_backbone(smooth_trilinear.curve)
    assert fit.shape == (4, 2)


def test_force_quadrilinear_over_trilinear_curve(smooth_trilinear):
    fit = fit_piecewise_backbone(smooth_trilinear.curve, mode="quadrilinear")
    assert fit.shape == (5, 2)


def test_force_trilinear_over_quadrilinear_curve(smooth_quadrilinear):
    fit = fit_piecewise_backbone(smooth_quadrilinear.curve, mode="trilinear")
    assert fit.shape == (4, 2)


def test_bilinear(bilinear_source):
    fit = fit_piecewise_backbone(bilinear_source.curve, mode="bilinear")
    assert fit.shape == (3, 2)
    assert np.allclose(fit[0], [0.0, 0.0])
    d_y, f_y = fit[1]
    d_end, _f_end = fit[2]
    assert d_y > 0 and f_y > 0
    assert d_end > d_y


def test_staircase_recovers_plateau(staircase):
    fit = fit_piecewise_backbone(staircase.curve)
    assert fit.shape == (5, 2)
    f_p = float(fit[3, 1])
    expected = staircase.expected_breaks
    assert expected is not None
    f_p_expected = float(expected[3, 1])
    assert _close(f_p, f_p_expected, rel=0.10), f"F_p={f_p}, expected~={f_p_expected}"


def test_peak_at_end_raises_for_tri_quadri_auto(peak_at_end):
    for mode in (None, "trilinear", "quadrilinear"):
        with pytest.raises(ValueError):
            fit_piecewise_backbone(peak_at_end.curve, mode=mode)


def test_contract_all_modes(smooth_quadrilinear):
    for mode in (None, "bilinear", "trilinear", "quadrilinear"):
        fit = fit_piecewise_backbone(smooth_quadrilinear.curve, mode=mode)
        assert fit.ndim == 2
        assert fit.shape[1] == 2
        assert np.allclose(fit[0], [0.0, 0.0])
        diffs = np.diff(fit[:, 0])
        assert (diffs >= -1e-9).all()
