"""Fit 4-segment piecewise-linear backbones to CBF-style pushover curves."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize_scalar

FloatArr = npt.NDArray[np.float64]

_ELASTIC_REF_FRACTION = 0.10


def fit_cbf_backbone(
    pushover: npt.ArrayLike,
    *,
    smoothing_window: int | None = None,
    yield_slope_fraction: float = 0.5,
    cusum_threshold: float = 4.0,
    cusum_drift: float = 0.5,
    cusum_search_start_factor: float = 3.0,
) -> FloatArr:
    """Fit a 4-segment backbone to a CBF-style pushover curve.

    Parameters
    ----------
    pushover
        Shape (N, 2): col 0 displacement (monotone), col 1 force.
    smoothing_window
        Rolling-mean window for yield-slope computation. Default: max(5, N // 100).
    yield_slope_fraction
        Fraction of initial tangent stiffness below which the yield kink is flagged.
    cusum_threshold
        CUSUM trigger threshold on standardised velocity.
    cusum_drift
        CUSUM drift parameter on standardised velocity.
    cusum_search_start_factor
        CUSUM begins searching at the first sample where d >= d_y * factor.
        Values > 1 are expected; this prevents the elastic→post-buckling kink
        from being mistaken for the fracture event.

    Returns
    -------
    (5, 2) float64 ndarray of breakpoints:
    [[0, 0], [d_y, f_y], [d_f, f_f], [d_r, f_r], [d_max, f_end]]
    """
    arr = np.asarray(pushover, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"pushover must be shape (N, 2); got {arr.shape}")
    if arr.shape[0] < 5:
        raise ValueError("pushover must have at least 5 points")

    d: FloatArr = arr[:, 0]
    f: FloatArr = arr[:, 1]
    n = len(d)

    i_peak = int(np.argmax(f))
    f_peak = float(f[i_peak])
    if f_peak <= 0.0:
        raise ValueError("Peak force must be positive")
    if i_peak >= n - 1:
        raise ValueError("No descending branch: peak is at the last data point")

    window = smoothing_window if smoothing_window is not None else max(5, n // 100)

    i_y, d_y_data, f_y_data = _detect_yield_point(d, f, i_peak, window, yield_slope_fraction)
    k_e = _cbf_elastic_slope(d, f, i_y, f_y_data)

    i_f, i_r = _detect_fracture_region(
        d, f, d_y_data, cusum_threshold, cusum_drift, cusum_search_start_factor,
    )

    d_f, f_f = float(d[i_f]), float(f[i_f])
    d_r, f_r = float(d[i_r]), float(f[i_r])
    d_max = float(d[-1])

    _k_pb, d_y, f_y, d_joint, f_joint = _fit_post_buckling(d, f, d_f, f_f, k_e)

    _k_r, f_end = _fit_residual(d, f, d_r, f_r)

    return np.asarray(
        [[0.0, 0.0], [d_y, f_y], [d_joint, f_joint], [d_r, f_r], [d_max, f_end]],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _detect_yield_point(
    d: FloatArr, f: FloatArr, i_peak: int, window: int, yield_slope_fraction: float
) -> tuple[int, float, float]:
    """Find the yield/buckling kink on the ascending branch.

    Returns (global_index, d_y, f_y). Falls back to the peak if no kink is found.
    """
    n_asc = i_peak + 1
    if n_asc < 4:
        return i_peak, float(d[i_peak]), float(f[i_peak])

    d_asc = d[:n_asc]
    f_asc = f[:n_asc]

    f_smooth = uniform_filter1d(f_asc.astype(np.float64), size=window)

    dd = np.diff(d_asc)
    dd = np.where(dd <= 0.0, 1e-12, dd)
    slopes = np.diff(f_smooth) / dd

    # Use the first n_init samples to estimate k_init.  Start the search only
    # AFTER n_init to avoid false triggers from smoothing boundary artefacts
    # (the reflected padding near index 0 can inflate f_smooth[0], making the
    # very first slope look artificially small).
    n_init = max(3, n_asc // 20)
    k_init = float(np.median(slopes[:n_init]))
    if k_init <= 0.0:
        return i_peak, float(d[i_peak]), float(f[i_peak])

    threshold = yield_slope_fraction * k_init

    for i in range(n_init, len(slopes)):
        if float(slopes[i]) < threshold:
            return i, float(d_asc[i]), float(f_asc[i])

    return i_peak, float(d[i_peak]), float(f[i_peak])


def _cbf_elastic_slope(d: FloatArr, f: FloatArr, i_y: int, f_y: float) -> float:
    """Secant stiffness at 10 % of the detected yield force."""
    f_ref = _ELASTIC_REF_FRACTION * f_y
    d_ref = float(np.interp(f_ref, f[: i_y + 1], d[: i_y + 1]))
    if d_ref <= 0.0:
        raise ValueError("Elastic reference displacement <= 0")
    return f_ref / d_ref


def _detect_fracture_region(
    d: FloatArr,
    f: FloatArr,
    d_y: float,
    cusum_threshold: float,
    cusum_drift: float,
    search_start_factor: float,
) -> tuple[int, int]:
    """Locate fracture endpoints via CUSUM on standardised velocity.

    Returns global indices (i_f, i_r) bounding the fracture event.
    """
    d_search_start = d_y * search_start_factor
    matches = np.flatnonzero(d >= d_search_start)
    if matches.size == 0:
        raise ValueError(
            f"CUSUM search start d_y*{search_start_factor} = {d_search_start:.4g} "
            f"exceeds d_max = {float(d[-1]):.4g}"
        )
    i_start = int(matches[0])

    d_s = d[i_start:]
    f_s = f[i_start:]
    if len(d_s) < 3:
        raise ValueError(
            f"Fewer than 3 samples remain after CUSUM search start (got {len(d_s)})"
        )

    velocity = np.gradient(f_s, d_s)
    mean = float(np.mean(velocity))
    std = float(np.std(velocity))
    if std <= 0.0:
        raise ValueError("Velocity has zero variance in CUSUM search window")
    s_data = (velocity - mean) / std

    n_s = len(s_data)
    s_pos = np.zeros(n_s)
    s_neg = np.zeros(n_s)
    drops: list[int] = []
    for i in range(1, n_s):
        s_pos[i] = max(0.0, s_pos[i - 1] + s_data[i] - cusum_drift)
        s_neg[i] = min(0.0, s_neg[i - 1] + s_data[i] + cusum_drift)
        if s_pos[i] > cusum_threshold:
            s_pos[i] = 0.0
        elif s_neg[i] < -cusum_threshold:
            drops.append(i)
            s_neg[i] = 0.0

    if not drops:
        raise ValueError(
            f"No fracture detected by CUSUM (threshold={cusum_threshold}, drift={cusum_drift})"
        )

    i_f = max(i_start, i_start + drops[0] - 1)
    i_r = min(i_start + drops[-1] + 1, len(d) - 1)
    return i_f, i_r


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def _area_error(d: FloatArr, f_exact: FloatArr, f_fit: FloatArr) -> float:
    return float(np.trapezoid(np.abs(f_exact - f_fit), d))


def _fit_post_buckling(
    d: FloatArr, f: FloatArr, d_f: float, f_f: float, k_e: float
) -> tuple[float, float, float, float, float]:
    """Post-buckling line pinned at (d_f, f_f); 1-D optimisation over slope k_pb."""
    k_pb_min = -abs(f_f) / max(d_f, 1e-9) * 10.0
    # Cap upper bound so d_y > 0 is guaranteed across the entire search domain.
    # d_y = (f_f - k_pb*d_f)/(k_e - k_pb) > 0  ⟺  k_pb < f_f/d_f (denom always > 0).
    k_pb_max = min(k_e * 0.999, f_f / max(d_f, 1e-9) * 0.999)

    def obj(k_pb: float) -> float:
        denom = k_e - k_pb
        if denom <= 0.0:
            return 1e18
        d_y = (f_f - k_pb * d_f) / denom
        if d_y <= 0.0 or d_y >= d_f:
            return 1e18
        mask = (d >= d_y) & (d <= d_f)
        if not bool(mask.any()):
            return 1e18
        return _area_error(d[mask], f[mask], f_f + k_pb * (d[mask] - d_f))

    if k_pb_max <= k_pb_min:
        # Degenerate: f_f is so low relative to d_f that no valid post-buckling slope exists.
        k_pb = 0.0
        d_y = f_f / max(k_e, 1e-9)
        return k_pb, d_y, k_e * d_y, d_f, f_f

    res = minimize_scalar(obj, bounds=(k_pb_min, k_pb_max), method="bounded")
    k_pb = float(res.x)
    denom = k_e - k_pb
    if denom <= 0.0:
        k_pb = 0.0
        denom = k_e
    d_y = (f_f - k_pb * d_f) / denom
    if d_y <= 0.0:
        d_y = f_f / max(k_e, 1e-9)
        k_pb = 0.0
    f_y = k_e * d_y
    return k_pb, d_y, f_y, d_f, f_f


def _fit_residual(
    d: FloatArr, f: FloatArr, d_r: float, f_r: float
) -> tuple[float, float]:
    """Residual line pinned at (d_r, f_r); 1-D optimisation with slope ≤ 0."""
    d_max = float(d[-1])
    span = d_max - d_r
    if span < 1e-9:
        return 0.0, f_r

    k_r_min = -f_r / span * 2.0

    def obj(k_r: float) -> float:
        mask = d >= d_r
        if not bool(mask.any()):
            return 1e18
        return _area_error(d[mask], f[mask], f_r + k_r * (d[mask] - d_r))

    res = minimize_scalar(obj, bounds=(k_r_min, 0.0), method="bounded")
    k_r = float(res.x)
    return k_r, f_r + k_r * span
