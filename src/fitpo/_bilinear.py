"""Bilinear backbone fits that end at a chosen point on the curve.

Both fits share the elastic stiffness rule of the De Luca fits (secant to a
fraction of the peak force) and differ only in where the second segment
ends. ``"bilinear-ultimate"`` runs to the last data point, or to the point
where the force first falls to zero for good, so it captures the whole
curve including any loss of strength. ``"bilinear-pctloss"`` stops where the
force has dropped to ``bilinear_end_fraction`` of the peak (0.80 by default,
i.e. a 20 % strength loss), a common definition of the usable deformation
capacity. In both, the yield force is
chosen to minimise the area between the fit and the data.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from fitpo._common import (
    FitResult,
    FloatArr,
    OriginMode,
    _area_error,
    _as_curve,
    _finish,
    _interval_area_error,
)
from fitpo._deluca import elastic_slope

type BilinearMode = Literal["bilinear-ultimate", "bilinear-pctloss"]
_BILINEAR_MODES: tuple[str, ...] = ("bilinear-ultimate", "bilinear-pctloss")


def fit_bilinear_backbone(
    pushover: npt.ArrayLike,
    *,
    mode: BilinearMode = "bilinear-ultimate",
    elastic_ref_fraction: float = 0.10,
    bilinear_end_fraction: float = 0.80,
    origin: OriginMode = "strict",
) -> FitResult:
    """Fit a bilinear backbone (elastic line + one post-yield line).

    Parameters
    ----------
    pushover
        ``(N, 2)`` curve: column 0 displacement (never decreasing), column 1
        force. Any consistent units work; the examples use mm
        and N.
    mode
        ``"bilinear-ultimate"`` (default): the second segment ends at the last
        data point, or at the last zero-force crossing when the curve ends
        below zero force. ``"bilinear-pctloss"``: the second segment ends where
        the force first drops to ``bilinear_end_fraction`` of the peak after
        the peak (or at the last point if it never does).
    elastic_ref_fraction
        The elastic stiffness is the secant to this fraction of the peak
        force on the rising branch.
    bilinear_end_fraction
        End-point force fraction for ``"bilinear-pctloss"``; unused otherwise.
    origin
        How to handle a curve whose first point is not ``(0, 0)``:

        - ``"strict"`` (default): raise a ``ValueError`` that says how far the
          first point is from the origin. Often the unloaded point ``(0, 0)``
          was simply not recorded, and no correction should be guessed.
        - ``"correct_disp"``: subtract ``d[0]`` from every displacement;
          forces are unchanged.
        - ``"correct_dispforc"``: subtract the whole first point
          ``(d[0], f[0])``.
        - ``"as-is"``: use the curve as given. The elastic segment still
          starts at the true origin, so any initial offset counts as elastic.

        With a correction, ``fitted_po`` is in the corrected coordinates.

    Returns
    -------
    FitResult
        ``fitted_po`` has 3 rows: origin, yield, end. ``objective`` is the
        area error over ``[d_y, d_end]`` (``d_end`` being ``d_u`` for
        ``"bilinear-ultimate"``).

    Warns
    -----
    UserWarning
        If an optimiser does not report convergence (``converged`` is then
        ``False``).

    Raises
    ------
    ValueError
        On invalid input, an unknown ``mode`` or a peak force that is not
        positive; if the zero-force crossing that ends ``"bilinear-ultimate"``
        cannot be located; or if the fitted shape fails the validity checks.

    Examples
    --------
    >>> import numpy as np
    >>> import fitpo
    >>> d = np.linspace(0.0, 120.0, 241)
    >>> f = np.interp(d, [0, 10, 40, 80, 120], [0, 1000, 1200, 600, 600])
    >>> curve = np.column_stack([d, f])
    >>> res = fitpo.fit_bilinear_backbone(curve)
    >>> print(np.round(res.fitted_po, 1))
    [[   0.     0. ]
     [  11.4 1144.6]
     [ 120.   600. ]]
    >>> res = fitpo.fit_bilinear_backbone(curve, mode="bilinear-pctloss")
    >>> print(np.round(res.fitted_po, 1))
    [[   0.     0. ]
     [  11.6 1160.5]
     [  56.   960. ]]
    """
    if mode not in _BILINEAR_MODES:
        raise ValueError(f"mode must be one of {_BILINEAR_MODES}; got {mode!r}")
    d, f = _as_curve(pushover, 3, origin=origin)

    i_peak = int(np.argmax(f))
    f_peak = float(f[i_peak])
    if f_peak <= 0.0:
        raise ValueError("Peak force must be positive")

    k_e = elastic_slope(d[: i_peak + 1], f[: i_peak + 1], f_peak, ref_fraction=elastic_ref_fraction)

    if mode == "bilinear-pctloss":
        d_end, f_end = _end_point_at_fraction(d, f, f_peak, i_peak, bilinear_end_fraction)
        f_y, success = _fit_yield_to_end(d, f, f_peak, k_e, d_end, f_end)
    else:
        d_end, f_end = _ultimate_point(d, f)
        f_y, success = _fit_yield_ultimate(d, f, f_peak, k_e, d_end, f_end)

    d_y = f_y / k_e
    bp = np.asarray([[0.0, 0.0], [d_y, f_y], [d_end, f_end]], dtype=np.float64)
    obj = _interval_area_error(d, f, bp, d_y, d_end)
    return _finish(bp, mode, obj, success, "fit_bilinear_backbone")


def _end_point_at_fraction(
    d: FloatArr, f: FloatArr, f_peak: float, i_peak: int, fraction: float
) -> tuple[float, float]:
    """First post-peak point where the force falls to ``fraction * f_peak``.

    Interpolates between the two samples either side of the crossing. Falls
    back to the last data point if the force never drops that far.
    """
    f_target = fraction * f_peak
    d_max = float(d[-1])
    for i in range(i_peak + 1, len(f)):
        if f[i] <= f_target:
            if f[i] == f[i - 1]:
                d_cross = float(d[i])
            else:
                t = (f_target - f[i - 1]) / (f[i] - f[i - 1])
                d_cross = float(d[i - 1] + t * (d[i] - d[i - 1]))
            d_end = min(d_cross, d_max)
            f_end = float(np.interp(d_end, d, f))
            return d_end, f_end
    return d_max, float(f[-1])


def _fit_yield_to_end(
    d: FloatArr, f: FloatArr, f_peak: float, k_e: float, d_end: float, f_end: float
) -> tuple[float, bool]:
    """Yield force minimising the area error over the data in ``[d_y, d_end]``."""

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
    return float(res.x), bool(res.success)


def _ultimate_point(d: FloatArr, f: FloatArr) -> tuple[float, float]:
    """End point for ``"bilinear-ultimate"``.

    The last data point if its force is non-negative; otherwise the last
    crossing from positive to negative force, so the fit never runs into
    negative force.
    """
    if f[-1] >= 0.0:
        return float(d[-1]), float(f[-1])
    sign = f >= 0.0
    transitions = np.flatnonzero(sign[:-1] & ~sign[1:])
    if transitions.size == 0:
        raise ValueError("Cannot locate zero-force crossing for ultimate point")
    i = int(transitions[-1])
    d_u = float(d[i] + (0.0 - f[i]) / (f[i + 1] - f[i]) * (d[i + 1] - d[i]))
    return d_u, 0.0


def _fit_yield_ultimate(
    d: FloatArr, f: FloatArr, f_peak: float, k_e: float, d_u: float, f_u: float
) -> tuple[float, bool]:
    """Yield force minimising the area error over ``[d_y, d_u]``.

    The yield force may exceed the peak (up to twice it), because a straight
    line to a low end point can need a high start to balance the area.
    """

    def obj(f_y: float) -> float:
        d_y = f_y / k_e
        if d_y <= 0.0 or d_y >= d_u:
            return 1e18
        mask = (d > d_y) & (d < d_u)
        d_inner = d[mask]
        f_inner = f[mask]
        d_grid = np.concatenate(([d_y], d_inner, [d_u]))
        f_exact_grid = np.concatenate(([float(np.interp(d_y, d, f))], f_inner, [f_u]))
        slope = (f_u - f_y) / (d_u - d_y)
        f_fit_grid = f_y + slope * (d_grid - d_y)
        return _area_error(d_grid, f_exact_grid, f_fit_grid)

    res = minimize_scalar(obj, bounds=(1e-9, 2.0 * f_peak), method="bounded")
    return float(res.x), bool(res.success)
