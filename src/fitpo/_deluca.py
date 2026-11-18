"""Trilinear and quadrilinear backbone fits following De Luca et al. (2013).

The fit splits a pushover curve at its peak. The elastic stiffness is the
secant to a small fraction of the peak force; a hardening line through the
peak is fitted to the rising branch; and a softening line from the peak,
optionally followed by a horizontal residual plateau, is fitted to the
falling branch. Each line is chosen to minimise the area between it and the
data. In ``"auto"`` mode the plateau is kept only when it clearly improves
the fit and is long and high enough to be meaningful.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, minimize_scalar

from fitpo._common import FitResult, FloatArr, _area_error, _as_curve, _interval_area_error

type DelucaMode = Literal["auto", "trilinear-deluca", "quadrilinear-deluca"]
_DELUCA_MODES: tuple[str, ...] = ("auto", "trilinear-deluca", "quadrilinear-deluca")


def fit_deluca_backbone(
    pushover: npt.ArrayLike,
    *,
    mode: DelucaMode = "auto",
    elastic_ref_fraction: float = 0.10,
    plateau_min_improvement: float = 0.05,
    plateau_min_force_fraction: float = 0.10,
    plateau_min_length_fraction: float = 0.05,
) -> FitResult:
    """Fit a trilinear or quadrilinear backbone to a pushover curve.

    Parameters
    ----------
    pushover
        ``(N, 2)`` curve: column 0 displacement (increasing, from 0),
        column 1 force.
    mode
        ``"auto"`` (default) fits both shapes and keeps the plateau only if
        the plateau-acceptance rule below holds. ``"trilinear-deluca"`` and
        ``"quadrilinear-deluca"`` force one shape.
    elastic_ref_fraction
        The elastic stiffness is the secant to this fraction of the peak
        force on the rising branch.
    plateau_min_improvement
        ``"auto"`` keeps the plateau only if it lowers the falling-branch
        area error by at least this fraction.
    plateau_min_force_fraction
        ``"auto"`` keeps the plateau only if its force is at least this
        fraction of the peak force.
    plateau_min_length_fraction
        ``"auto"`` keeps the plateau only if it covers at least this
        fraction of the falling branch.

    Returns
    -------
    FitResult
        ``fitted_po`` has 4 rows (trilinear: origin, yield, peak, end) or
        5 rows (quadrilinear: origin, yield, peak, plateau start, end).
        ``objective`` is the area error over ``[d_y, d_max]``.
    """
    if mode not in _DELUCA_MODES:
        raise ValueError(f"mode must be one of {_DELUCA_MODES}; got {mode!r}")
    d, f = _as_curve(pushover, 3)

    i_peak = int(np.argmax(f))
    d_peak = float(d[i_peak])
    f_peak = float(f[i_peak])
    if f_peak <= 0.0:
        raise ValueError("Peak force must be positive")

    k_e = elastic_slope(d[: i_peak + 1], f[: i_peak + 1], f_peak, ref_fraction=elastic_ref_fraction)

    if i_peak >= len(f) - 1:
        raise ValueError("No descending branch: peak is at the last data point")

    # Success flags of every optimiser call, so `converged` covers all of them.
    ok: list[bool] = []
    _k_h, d_y, f_y = _fit_hardening(d, f, d_peak, f_peak, k_e, i_peak, ok)
    d_max = float(d[-1])

    def tri(k_s: float) -> FitResult:
        f_end = f_peak + k_s * (d_max - d_peak)
        bp = np.asarray(
            [[0.0, 0.0], [d_y, f_y], [d_peak, f_peak], [d_max, f_end]], dtype=np.float64
        )
        return _result(d, f, bp, "trilinear-deluca", d_y, d_max, ok)

    def quad(f_p: float, d_b: float) -> FitResult:
        bp = np.asarray(
            [[0.0, 0.0], [d_y, f_y], [d_peak, f_peak], [d_b, f_p], [d_max, f_p]],
            dtype=np.float64,
        )
        return _result(d, f, bp, "quadrilinear-deluca", d_y, d_max, ok)

    if mode == "trilinear-deluca":
        return tri(_fit_softening(d, f, d_peak, f_peak, i_peak, ok))

    if mode == "quadrilinear-deluca":
        _k_s, f_p, d_b = _fit_softening_plateau(
            d, f, d_peak, f_peak, i_peak, plateau_min_force_fraction, ok
        )
        return quad(f_p, d_b)

    # Auto mode: fit the falling branch both ways and compare their area errors
    # over the falling branch only (the rising branch is the same for both).
    k_s_tri = _fit_softening(d, f, d_peak, f_peak, i_peak, ok)
    d_desc: FloatArr = d[i_peak:]
    f_desc: FloatArr = f[i_peak:]
    f_fit_tri = f_peak + k_s_tri * (d_desc - d_peak)
    a3 = _area_error(d_desc, f_desc, f_fit_tri)

    k_s_q, f_p_q, d_b_q = _fit_softening_plateau(
        d, f, d_peak, f_peak, i_peak, plateau_min_force_fraction, ok
    )
    f_fit_q = np.where(d_desc <= d_b_q, f_peak + k_s_q * (d_desc - d_peak), f_p_q)
    a4 = _area_error(d_desc, f_desc, f_fit_q)

    # Plateau-acceptance rule: a plateau adds two parameters, so keep it only
    # if it pays for them with a clear improvement, and if it is neither a
    # near-zero force level nor a short stub at the end of the curve.
    plateau_length = d_max - d_b_q
    descending_length = d_max - d_peak
    accept_plateau = (
        a4 < (1.0 - plateau_min_improvement) * a3
        and f_p_q >= plateau_min_force_fraction * f_peak
        and plateau_length >= plateau_min_length_fraction * descending_length
    )
    if accept_plateau:
        return quad(f_p_q, d_b_q)
    return tri(k_s_tri)


def elastic_slope(
    d_asc: FloatArr, f_asc: FloatArr, f_peak: float, *, ref_fraction: float = 0.10
) -> float:
    """Secant stiffness to ``ref_fraction * f_peak`` on the rising branch."""
    f_ref = ref_fraction * f_peak
    d_ref = float(np.interp(f_ref, f_asc, d_asc))
    if d_ref <= 0.0:
        raise ValueError("Elastic reference displacement <= 0")
    return f_ref / d_ref


def _result(
    d: FloatArr,
    f: FloatArr,
    bp: FloatArr,
    mode: str,
    d_lo: float,
    d_hi: float,
    ok: list[bool],
) -> FitResult:
    """Package breakpoints with the area error over ``[d_lo, d_hi]``."""
    return FitResult(bp, mode, _interval_area_error(d, f, bp, d_lo, d_hi), all(ok))


def _fit_hardening(
    d: FloatArr,
    f: FloatArr,
    d_peak: float,
    f_peak: float,
    k_e: float,
    i_peak: int,
    ok: list[bool],
) -> tuple[float, float, float]:
    """Fit the hardening line through the peak; return ``(k_h, d_y, f_y)``.

    The yield point is where the hardening line meets the elastic line
    ``F = k_e d``, so only the hardening slope ``k_h`` is free.
    """
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
    ok.append(bool(res.success))
    k_h = float(res.x)
    denom = k_e - k_h
    d_y = (f_peak - k_h * d_peak) / denom
    f_y = k_e * d_y
    return k_h, float(d_y), float(f_y)


def _fit_softening(
    d: FloatArr,
    f: FloatArr,
    d_peak: float,
    f_peak: float,
    i_peak: int,
    ok: list[bool],
) -> float:
    """Fit a single softening line from the peak to the end; return its slope."""
    d_desc: FloatArr = d[i_peak:]
    f_desc: FloatArr = f[i_peak:]
    d_max = float(d_desc[-1])
    # Steepest slope searched: the line would reach zero force ten times
    # faster than over the whole falling branch.
    k_s_min = -f_peak / max(d_max - d_peak, 1e-9) * 10.0

    def obj(k_s: float) -> float:
        f_fit = f_peak + k_s * (d_desc - d_peak)
        return _area_error(d_desc, f_desc, f_fit)

    res = minimize_scalar(obj, bounds=(k_s_min, 0.0), method="bounded")
    ok.append(bool(res.success))
    return float(res.x)


def _fit_softening_plateau(
    d: FloatArr,
    f: FloatArr,
    d_peak: float,
    f_peak: float,
    i_peak: int,
    plateau_min_force_fraction: float,
    ok: list[bool],
) -> tuple[float, float, float]:
    """Fit softening line + horizontal plateau; return ``(k_s, f_p, d_b)``."""
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

    # Start from the single-line fit and a plateau just above the acceptance
    # floor, so the optimiser begins from a feasible shape.
    k_s_init = _fit_softening(d, f, d_peak, f_peak, i_peak, ok)
    f_p_init = max(float(f_desc[-1]), plateau_min_force_fraction * f_peak * 1.1)
    f_p_init = min(f_p_init, f_peak * 0.99)

    res = minimize(
        obj,
        x0=np.asarray([k_s_init, f_p_init], dtype=np.float64),
        bounds=[(k_s_min, -1e-9), (1e-9, f_peak * 0.999)],
        method="L-BFGS-B",
    )
    ok.append(bool(res.success))
    k_s = float(res.x[0])
    f_p = float(res.x[1])
    d_b = d_peak + (f_p - f_peak) / k_s
    return k_s, f_p, float(d_b)
