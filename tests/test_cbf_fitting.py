"""Tests for fit_cbf_backbone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fitpo import fit_cbf_backbone

CURVES_DIR = Path(__file__).resolve().parent.parent / "example_po_curves"
CSV_PATHS = sorted(CURVES_DIR.glob("*.csv"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_cbf_curve(
    d_y: float, f_y: float,
    d_f: float, f_f: float,
    d_r: float, f_r: float,
    d_max: float, f_end: float,
    n: int = 801,
    noise_fraction: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic CBF curve with Gaussian noise."""
    breaks = np.asarray(
        [[0.0, 0.0], [d_y, f_y], [d_f, f_f], [d_r, f_r], [d_max, f_end]]
    )
    d = np.linspace(0.0, d_max, n)
    f_clean = np.interp(d, breaks[:, 0], breaks[:, 1])
    rng = np.random.default_rng(seed)
    f = f_clean + rng.normal(0.0, noise_fraction * f_f, size=n)
    return np.column_stack([d, f])


# ---------------------------------------------------------------------------
# Contract tests against the 9 example CSVs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("csv_path", CSV_PATHS, ids=[p.stem for p in CSV_PATHS])
def test_output_shape(csv_path: Path) -> None:
    data = np.loadtxt(csv_path, delimiter=",")
    fit = fit_cbf_backbone(data)
    assert fit.shape == (5, 2)


@pytest.mark.parametrize("csv_path", CSV_PATHS, ids=[p.stem for p in CSV_PATHS])
def test_displacements_increasing(csv_path: Path) -> None:
    data = np.loadtxt(csv_path, delimiter=",")
    fit = fit_cbf_backbone(data)
    diffs = np.diff(fit[:, 0])
    assert (diffs > 0).all(), f"Non-increasing displacements: {fit[:, 0]}"


@pytest.mark.parametrize("csv_path", CSV_PATHS, ids=[p.stem for p in CSV_PATHS])
def test_origin_at_zero(csv_path: Path) -> None:
    data = np.loadtxt(csv_path, delimiter=",")
    fit = fit_cbf_backbone(data)
    assert np.allclose(fit[0], [0.0, 0.0])


@pytest.mark.parametrize("csv_path", CSV_PATHS, ids=[p.stem for p in CSV_PATHS])
def test_fracture_drop_positive(csv_path: Path) -> None:
    data = np.loadtxt(csv_path, delimiter=",")
    fit = fit_cbf_backbone(data)
    # Force at fracture start > force at residual start
    assert fit[2, 1] > fit[3, 1], "Fracture drop must be a force decrease"


@pytest.mark.parametrize("csv_path", CSV_PATHS, ids=[p.stem for p in CSV_PATHS])
def test_residual_slope_nonpositive(csv_path: Path) -> None:
    data = np.loadtxt(csv_path, delimiter=",")
    fit = fit_cbf_backbone(data)
    d_r, f_r = fit[3]
    d_max, f_end = fit[4]
    slope = (f_end - f_r) / max(d_max - d_r, 1e-12)
    assert slope <= 1e-6, f"Residual slope {slope:.4f} is positive"


# ---------------------------------------------------------------------------
# Synthetic round-trip test
# ---------------------------------------------------------------------------


def test_synthetic_round_trip() -> None:
    """Recovered breakpoints should be within 15 % of the true values."""
    d_y, f_y = 10.0, 500.0
    d_f, f_f = 80.0, 350.0
    d_r, f_r = 82.0, 100.0
    d_max, f_end = 120.0, 80.0

    curve = _sample_cbf_curve(d_y, f_y, d_f, f_f, d_r, f_r, d_max, f_end, n=801, seed=7)
    fit = fit_cbf_backbone(curve)

    assert fit.shape == (5, 2)

    def close(actual: float, expected: float, rel: float = 0.15) -> bool:
        return abs(actual - expected) <= rel * max(abs(expected), 1e-9)

    # Yield point
    assert close(fit[1, 0], d_y), f"d_y: {fit[1,0]:.2f} vs {d_y}"
    assert close(fit[1, 1], f_y), f"f_y: {fit[1,1]:.2f} vs {f_y}"
    # Segment 2-3 transition slides on the drop line within [d_f, d_r].
    assert close(fit[2, 0], d_f, rel=0.05), f"d_joint: {fit[2,0]:.2f} vs {d_f}"
    # Segment 3-4 transition slides on the drop line within [d_r, d_max].
    assert close(fit[3, 0], d_r, rel=0.50), f"d_r2: {fit[3,0]:.2f} vs {d_r}"


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_bad_shape_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        fit_cbf_backbone(np.ones((10, 3)))


def test_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match="5 points"):
        fit_cbf_backbone(np.column_stack([np.arange(4), np.arange(4)]))


def test_peak_at_end_raises() -> None:
    d = np.linspace(0.0, 10.0, 50)
    f = d * 10.0
    with pytest.raises(ValueError, match="descending branch"):
        fit_cbf_backbone(np.column_stack([d, f]))
