"""Shared pieces for all fitpo fits: the result type and the area-error measures.

Every backbone function returns a :class:`FitResult`, and every fit is scored
by the same kind of measure: the area between the data and the fitted
piecewise-linear curve. This module holds the single copy of that measure,
the public :func:`area_error` helper built on it, and the small input
reshaping step all fitting functions share.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.float64]


class FitResult(NamedTuple):
    """Result of a backbone fit.

    Attributes
    ----------
    fitted_po
        ``(M, 2)`` breakpoints of the fitted curve: column 0 displacement,
        column 1 force. The number of rows depends on the mode, so always use
        ``len(fitted_po)`` rather than assuming a row count.
    mode
        The shape that was actually fitted, e.g. ``"trilinear-deluca"``.
    objective
        Area error between the data and the returned shape, over the range
        the mode is optimised on (see each function's docstring).
    converged
        ``True`` when every optimiser call made during the fit reported
        success.
    """

    fitted_po: FloatArr
    mode: str
    objective: float
    converged: bool


def _area_error(d: FloatArr, f_exact: FloatArr, f_fit: FloatArr) -> float:
    """Trapezoidal integral of ``|f_exact - f_fit|`` over the grid ``d``."""
    return float(np.trapezoid(np.abs(f_exact - f_fit), d))


def _as_curve(pushover: npt.ArrayLike, min_points: int) -> tuple[FloatArr, FloatArr]:
    """Check the ``(N, 2)`` shape and split a curve into displacement and force."""
    arr = np.asarray(pushover, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"pushover must be shape (N, 2); got {arr.shape}")
    if arr.shape[0] < min_points:
        raise ValueError(f"pushover must have at least {min_points} points")
    return arr[:, 0], arr[:, 1]


def _interval_area_error(
    d: FloatArr, f: FloatArr, fitted_po: FloatArr, d_lo: float, d_hi: float
) -> float:
    """Area between the data and a fitted shape over ``[d_lo, d_hi]``.

    Both curves are piecewise linear, so integrating ``|F - F_hat|`` with the
    trapezoid rule is exact up to sign changes between grid points. The grid
    therefore holds every data point and every fitted breakpoint inside the
    range, plus the two range ends, so neither curve has a kink between grid
    points.
    """
    if d_hi <= d_lo:
        return 0.0
    bp = fitted_po[:, 0]
    grid = np.unique(
        np.concatenate(
            ([d_lo, d_hi], d[(d > d_lo) & (d < d_hi)], bp[(bp > d_lo) & (bp < d_hi)])
        )
    )
    f_data = np.interp(grid, d, f)
    f_fit = np.interp(grid, fitted_po[:, 0], fitted_po[:, 1])
    return _area_error(grid, f_data, f_fit)


def area_error(
    curve: npt.ArrayLike, fitted_po: npt.ArrayLike, *, normalised: bool = False
) -> float:
    r"""Area between a pushover curve and a fitted shape, over the whole curve.

    Computes

    .. math::

        A = \int_{d_0}^{d_{max}} \left| F(d) - \hat F(d) \right| \, \mathrm{d}d

    where :math:`F` is the curve, :math:`\hat F` is interpolated linearly from
    ``fitted_po``, and :math:`[d_0, d_{max}]` is the curve's displacement
    range (:math:`[0, d_{max}]` for a curve that starts at the origin).

    Unlike ``FitResult.objective``, whose range depends on the mode, this
    uses one range for every fit, so fits of different shapes or from
    different functions can be compared.

    Parameters
    ----------
    curve
        ``(N, 2)`` pushover curve: column 0 displacement, column 1 force.
    fitted_po
        ``(M, 2)`` breakpoints of the fitted curve, e.g. ``res.fitted_po``.
    normalised
        If ``True``, divide by :math:`\int |F| \, \mathrm{d}d` over the same
        range, which makes the result dimensionless.

    Returns
    -------
    float
        The area error, in force times displacement units, or dimensionless
        when ``normalised`` is ``True``.
    """
    d, f = _as_curve(curve, 2)
    bp = np.asarray(fitted_po, dtype=np.float64)
    if bp.ndim != 2 or bp.shape[1] != 2:
        raise ValueError(f"fitted_po must be shape (M, 2); got {bp.shape}")
    err = _interval_area_error(d, f, bp, float(d[0]), float(d[-1]))
    if normalised:
        return err / _area_error(d, f, np.zeros_like(f))
    return err
