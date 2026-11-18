"""Shared pieces for all fitpo fits: the result type and the area-error measures.

Every backbone function returns a :class:`FitResult`, and every fit is scored
by the same kind of measure: the area between the data and the fitted
piecewise-linear curve. This module holds the single copy of that measure,
the public :func:`area_error` helper built on it, and the small input
reshaping step all fitting functions share.
"""

from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.float64]

# How a curve whose first point is not (0, 0) is handled:
# - "strict": refuse the curve, with an error saying how far out it is;
# - "correct_disp": subtract d[0] from every displacement (forces unchanged);
# - "correct_dispforc": subtract the whole first point (d[0], f[0]);
# - "as-is": use the curve as given; the elastic line still starts at (0, 0).
type OriginMode = Literal["strict", "correct_disp", "correct_dispforc", "as-is"]
_ORIGIN_MODES: tuple[str, ...] = ("strict", "correct_disp", "correct_dispforc", "as-is")


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


def _check_finite(arr: FloatArr, name: str) -> None:
    """Raise if ``arr`` holds NaN or inf, which would silently corrupt every fit."""
    if not bool(np.isfinite(arr).all()):
        raise ValueError(f"{name} contains NaN or inf values")


def _check_origin(origin: str) -> None:
    """Raise on an unknown ``origin`` value, before any data is touched."""
    if origin not in _ORIGIN_MODES:
        raise ValueError(f"origin must be one of {_ORIGIN_MODES}; got {origin!r}")


def _as_curve(
    pushover: npt.ArrayLike,
    min_points: int,
    *,
    name: str = "pushover",
    monotone: bool = True,
    origin: OriginMode | None = None,
) -> tuple[FloatArr, FloatArr]:
    """Check a curve and split it into displacement and force.

    Checks the ``(N, 2)`` shape, the point count and that every value is
    finite. With ``monotone=True`` (all backbone fits), displacement must
    never decrease; repeated values are allowed because pushover analyses
    can record several steps at the same displacement.

    ``origin`` sets how a first point other than ``(0, 0)`` is handled (see
    :data:`OriginMode`); ``None`` skips origin handling, for data such as
    cyclic traces or fitted shapes where it doesn't apply.
    """
    arr = np.asarray(pushover, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be shape (N, 2); got {arr.shape}")
    if arr.shape[0] < min_points:
        raise ValueError(f"{name} must have at least {min_points} points")
    _check_finite(arr, name)
    d = arr[:, 0]
    f = arr[:, 1]
    if monotone:
        steps = np.diff(d)
        if bool((steps < 0.0).any()):
            i = int(np.flatnonzero(steps < 0.0)[0])
            raise ValueError(
                f"{name} displacement must never decrease; d[{i + 1}] = {d[i + 1]:.6g} "
                f"< d[{i}] = {d[i]:.6g}"
            )
    if origin is None or origin == "as-is":
        return d, f
    _check_origin(origin)
    if origin == "strict":
        # A curve that doesn't start at (0, 0) is often missing its first,
        # unloaded point. Refuse rather than guess, and say how far out it is.
        if d[0] != 0.0 or f[0] != 0.0:
            d_rel = abs(d[0]) / max(float(np.max(np.abs(d))), 1e-300)
            f_rel = abs(f[0]) / max(float(np.max(np.abs(f))), 1e-300)
            raise ValueError(
                f"{name} must start at (0, 0) with origin='strict'; the first point is "
                f"({d[0]:.6g}, {f[0]:.6g}), i.e. {100 * d_rel:.3g} % of the largest "
                f"|displacement| and {100 * f_rel:.3g} % of the largest |force|. Add the "
                "unloaded point (0, 0) if it is missing, or choose origin='correct_disp', "
                "'correct_dispforc' or 'as-is'"
            )
        return d, f
    d = d - d[0]
    if origin == "correct_dispforc":
        f = f - f[0]
    return d, f


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
    curve: npt.ArrayLike,
    fitted_po: npt.ArrayLike,
    *,
    normalised: bool = False,
    origin: OriginMode = "strict",
) -> float:
    r"""Area between a pushover curve and a fitted shape, over the whole curve.

    Computes

    .. math::

        A = \int_{d_0}^{d_{max}} \left| F(d) - \hat F(d) \right| \, \mathrm{d}d

    where :math:`F` is the curve, :math:`\hat F` is interpolated linearly from
    ``fitted_po``, and :math:`[d_0, d_{max}]` is the curve's displacement
    range. Every ``origin`` setting except ``"as-is"`` makes the curve start
    at ``d = 0``, so the range is then :math:`[0, d_{max}]`.

    Unlike ``FitResult.objective``, whose range depends on the mode, this
    uses one range for every fit, so fits of different shapes or from
    different functions can be compared.

    Parameters
    ----------
    curve
        ``(N, 2)`` pushover curve: column 0 displacement (never decreasing),
        column 1 force.
    fitted_po
        ``(M, 2)`` breakpoints of the fitted curve, e.g. ``res.fitted_po``.
    normalised
        If ``True``, divide by :math:`\int |F| \, \mathrm{d}d` over the same
        range, which makes the result dimensionless.
    origin
        How to handle a curve whose first point is not ``(0, 0)``, with the
        same values as the backbone functions: ``"strict"`` (default) raises,
        ``"correct_disp"`` subtracts ``d[0]``, ``"correct_dispforc"``
        subtracts ``(d[0], f[0])``, and ``"as-is"`` uses the curve as given.
        Use the same setting as the fit that produced ``fitted_po``, so the
        curve and the fit are in the same coordinates.

    Returns
    -------
    float
        The area error, in force times displacement units, or dimensionless
        when ``normalised`` is ``True``.

    Raises
    ------
    ValueError
        If either array has the wrong shape or holds NaN or inf, if a
        displacement column ever decreases, if ``origin`` is unknown, or if
        ``origin="strict"`` and the curve does not start at ``(0, 0)``.
    """
    d, f = _as_curve(curve, 2, name="curve", origin=origin)
    # The fitted shape is checked too: np.interp gives meaningless values for
    # breakpoints whose displacement goes backwards.
    bp = np.column_stack(_as_curve(fitted_po, 2, name="fitted_po"))
    err = _interval_area_error(d, f, bp, float(d[0]), float(d[-1]))
    if normalised:
        return err / _area_error(d, f, np.zeros_like(f))
    return err
