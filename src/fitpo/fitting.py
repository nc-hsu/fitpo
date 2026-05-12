"""Fit piecewise-linear backbones to SDOF pushover curves."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, minimize_scalar

Mode = Literal["bilinear", "bilinear_incl_neg_stiffness", "trilinear", "quadrilinear"]
FloatArr = npt.NDArray[np.float64]

ELASTIC_REF_FRACTION = 0.10
BILINEAR_END_FRACTION = 0.80
TOL_IMPROVEMENT = 0.05
PLATEAU_FORCE_FLOOR = 0.10
PLATEAU_LENGTH_FLOOR = 0.05


def fit_piecewise_backbone(
    pushover: npt.ArrayLike,
    mode: Mode | None = None,
) -> FloatArr:
    """Fit a piecewise-linear backbone to an SDOF pushover curve.

    Parameters
    ----------
    pushover
        Shape (N, 2): col 0 displacement (monotone, from 0), col 1 force.
    mode
        None (default) = auto trilinear or quadrilinear per rule-3 test.
        Otherwise force bilinear / bilinear_incl_neg_stiffness / trilinear /
        quadrilinear. ``bilinear_incl_neg_stiffness`` anchors segment 2 to the
        last data point, or to the zero-force crossing when the curve dips
        negative.

    Returns
    -------
    (M, 2) float64 ndarray of breakpoints including origin and endpoint.
    """
    arr = np.asarray(pushover, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"pushover must be shape (N, 2); got {arr.shape}")
    if arr.shape[0] < 3:
        raise ValueError("pushover must have at least 3 points")
    d: FloatArr = arr[:, 0]
    f: FloatArr = arr[:, 1]

    i_peak = int(np.argmax(f))
    d_peak = float(d[i_peak])
    f_peak = float(f[i_peak])
    if f_peak <= 0.0:
        raise ValueError("Peak force must be positive")

    k_e = _elastic_slope(d[: i_peak + 1], f[: i_peak + 1], f_peak)

    if mode == "bilinear":
        return _fit_bilinear(d, f, d_peak, f_peak, k_e, i_peak)

    if mode == "bilinear_incl_neg_stiffness":
        return _fit_bilinear_incl_neg_stiffness(d, f, f_peak, k_e)

    if i_peak >= len(f) - 1:
        raise ValueError("No descending branch: peak is at the last data point")

    k_h, d_y, f_y = _fit_hardening(d, f, d_peak, f_peak, k_e, i_peak)
    d_max = float(d[-1])

    if mode == "trilinear":
        k_s = _fit_softening(d, f, d_peak, f_peak, i_peak)
        f_end = f_peak + k_s * (d_max - d_peak)
        return np.asarray(
            [[0.0, 0.0], [d_y, f_y], [d_peak, f_peak], [d_max, f_end]],
            dtype=np.float64,
        )

    if mode == "quadrilinear":
        k_s, f_p, d_b = _fit_softening_plateau(d, f, d_peak, f_peak, i_peak)
        return np.asarray(
            [[0.0, 0.0], [d_y, f_y], [d_peak, f_peak], [d_b, f_p], [d_max, f_p]],
            dtype=np.float64,
        )

    # Auto mode: compare trilinear vs quadrilinear softening fits
    k_s_tri = _fit_softening(d, f, d_peak, f_peak, i_peak)
    d_desc: FloatArr = d[i_peak:]
    f_desc: FloatArr = f[i_peak:]
    f_fit_tri = f_peak + k_s_tri * (d_desc - d_peak)
    a3 = _area_error(d_desc, f_desc, f_fit_tri)

    k_s_q, f_p_q, d_b_q = _fit_softening_plateau(d, f, d_peak, f_peak, i_peak)
    f_fit_q = np.where(d_desc <= d_b_q, f_peak + k_s_q * (d_desc - d_peak), f_p_q)
    a4 = _area_error(d_desc, f_desc, f_fit_q)

    plateau_length = d_max - d_b_q
    descending_length = d_max - d_peak
    accept_plateau = (
        a4 < (1.0 - TOL_IMPROVEMENT) * a3
        and f_p_q >= PLATEAU_FORCE_FLOOR * f_peak
        and plateau_length >= PLATEAU_LENGTH_FLOOR * descending_length
    )

    if accept_plateau:
        return np.asarray(
            [[0.0, 0.0], [d_y, f_y], [d_peak, f_peak], [d_b_q, f_p_q], [d_max, f_p_q]],
            dtype=np.float64,
        )
    f_end = f_peak + k_s_tri * (d_max - d_peak)
    return np.asarray(
        [[0.0, 0.0], [d_y, f_y], [d_peak, f_peak], [d_max, f_end]],
        dtype=np.float64,
    )


def _elastic_slope(d_asc: FloatArr, f_asc: FloatArr, f_peak: float) -> float:
    f_ref = ELASTIC_REF_FRACTION * f_peak
    d_ref = float(np.interp(f_ref, f_asc, d_asc))
    if d_ref <= 0.0:
        raise ValueError("Elastic reference displacement <= 0")
    return f_ref / d_ref


def _bilinear_end_point(
    d: FloatArr, f: FloatArr, f_peak: float, i_peak: int
) -> tuple[float, float]:
    f_target = BILINEAR_END_FRACTION * f_peak
    d_max = float(d[-1])
    for i in range(i_peak + 1, len(f)):
        if f[i] <= f_target:
            if f[i] == f[i - 1]:
                d_80 = float(d[i])
            else:
                t = (f_target - f[i - 1]) / (f[i] - f[i - 1])
                d_80 = float(d[i - 1] + t * (d[i] - d[i - 1]))
            d_end = min(d_80, d_max)
            f_end = float(np.interp(d_end, d, f))
            return d_end, f_end
    return d_max, float(f[-1])


def _fit_bilinear(
    d: FloatArr, f: FloatArr, d_peak: float, f_peak: float, k_e: float, i_peak: int
) -> FloatArr:
    d_end, f_end = _bilinear_end_point(d, f, f_peak, i_peak)

    def obj(f_y: float) -> float:
        d_y = f_y / k_e
        if d_y <= 0.0 or d_y >= d_end:
            return 1e18
        slope = (f_end - f_y) / (d_end - d_y)
        mask = (d >= d_y) & (d <= d_end)
        if not bool(mask.any()):
            return 1e18
        d_seg = d[mask]
        f_seg = f[mask]
        f_fit_seg = f_y + slope * (d_seg - d_y)
        return _area_error(d_seg, f_seg, f_fit_seg)

    res = minimize_scalar(obj, bounds=(1e-9, f_peak * 0.999), method="bounded")
    f_y = float(res.x)
    d_y = f_y / k_e
    return np.asarray([[0.0, 0.0], [d_y, f_y], [d_end, f_end]], dtype=np.float64)


def _ultimate_point(d: FloatArr, f: FloatArr) -> tuple[float, float]:
    if f[-1] >= 0.0:
        return float(d[-1]), float(f[-1])
    sign = f >= 0.0
    transitions = np.flatnonzero(sign[:-1] & ~sign[1:])
    if transitions.size == 0:
        raise ValueError("Cannot locate zero-force crossing for ultimate point")
    i = int(transitions[-1])
    d_u = float(d[i] + (0.0 - f[i]) / (f[i + 1] - f[i]) * (d[i + 1] - d[i]))
    return d_u, 0.0


def _fit_bilinear_incl_neg_stiffness(
    d: FloatArr, f: FloatArr, f_peak: float, k_e: float
) -> FloatArr:
    d_u, f_u = _ultimate_point(d, f)

    def obj(f_y: float) -> float:
        d_y = f_y / k_e
        if d_y <= 0.0 or d_y >= d_u:
            return 1e18
        mask = (d > d_y) & (d < d_u)
        d_inner = d[mask]
        f_inner = f[mask]
        d_grid = np.concatenate(([d_y], d_inner, [d_u]))
        f_exact_grid = np.concatenate(
            ([float(np.interp(d_y, d, f))], f_inner, [f_u])
        )
        slope = (f_u - f_y) / (d_u - d_y)
        f_fit_grid = f_y + slope * (d_grid - d_y)
        return _area_error(d_grid, f_exact_grid, f_fit_grid)

    res = minimize_scalar(obj, bounds=(1e-9, 2.0 * f_peak), method="bounded")
    f_y = float(res.x)
    d_y = f_y / k_e
    return np.asarray([[0.0, 0.0], [d_y, f_y], [d_u, f_u]], dtype=np.float64)


def _fit_hardening(
    d: FloatArr, f: FloatArr, d_peak: float, f_peak: float, k_e: float, i_peak: int
) -> tuple[float, float, float]:
    d_asc = d[: i_peak + 1]
    f_asc = f[: i_peak + 1]

    def obj(k_h: float) -> float:
        denom = k_e - k_h
        if denom <= 0.0:
            return 1e18
        d_y = (f_peak - k_h * d_peak) / denom
        if d_y <= 0.0 or d_y >= d_peak:
            return 1e18
        mask = (d_asc >= d_y) & (d_asc <= d_peak)
        if not bool(mask.any()):
            return 1e18
        d_seg = d_asc[mask]
        f_seg = f_asc[mask]
        f_fit_seg = f_peak + k_h * (d_seg - d_peak)
        return _area_error(d_seg, f_seg, f_fit_seg)

    res = minimize_scalar(obj, bounds=(0.0, k_e * 0.999), method="bounded")
    k_h = float(res.x)
    denom = k_e - k_h
    d_y = (f_peak - k_h * d_peak) / denom
    f_y = k_e * d_y
    return k_h, float(d_y), float(f_y)


def _fit_softening(
    d: FloatArr, f: FloatArr, d_peak: float, f_peak: float, i_peak: int
) -> float:
    d_desc: FloatArr = d[i_peak:]
    f_desc: FloatArr = f[i_peak:]
    d_max = float(d_desc[-1])
    k_s_min = -f_peak / max(d_max - d_peak, 1e-9) * 10.0

    def obj(k_s: float) -> float:
        f_fit = f_peak + k_s * (d_desc - d_peak)
        return _area_error(d_desc, f_desc, f_fit)

    res = minimize_scalar(obj, bounds=(k_s_min, 0.0), method="bounded")
    return float(res.x)


def _fit_softening_plateau(
    d: FloatArr, f: FloatArr, d_peak: float, f_peak: float, i_peak: int
) -> tuple[float, float, float]:
    d_desc: FloatArr = d[i_peak:]
    f_desc: FloatArr = f[i_peak:]
    d_max = float(d_desc[-1])
    k_s_min = -f_peak / max(d_max - d_peak, 1e-9) * 10.0

    def obj(x: FloatArr) -> float:
        k_s = float(x[0])
        f_p = float(x[1])
        if k_s >= 0.0 or f_p <= 0.0 or f_p >= f_peak:
            return 1e18
        d_b = d_peak + (f_p - f_peak) / k_s
        if d_b <= d_peak or d_b >= d_max:
            return 1e18
        f_fit = np.where(
            d_desc <= d_b,
            f_peak + k_s * (d_desc - d_peak),
            f_p,
        )
        return _area_error(d_desc, f_desc, f_fit)

    k_s_init = _fit_softening(d, f, d_peak, f_peak, i_peak)
    f_p_init = max(float(f_desc[-1]), PLATEAU_FORCE_FLOOR * f_peak * 1.1)
    f_p_init = min(f_p_init, f_peak * 0.99)

    res = minimize(
        obj,
        x0=np.asarray([k_s_init, f_p_init], dtype=np.float64),
        bounds=[(k_s_min, -1e-9), (1e-9, f_peak * 0.999)],
        method="L-BFGS-B",
    )
    k_s = float(res.x[0])
    f_p = float(res.x[1])
    d_b = d_peak + (f_p - f_peak) / k_s
    return k_s, f_p, float(d_b)


def _area_error(d: FloatArr, f_exact: FloatArr, f_fit: FloatArr) -> float:
    return float(np.trapezoid(np.abs(f_exact - f_fit), d))
