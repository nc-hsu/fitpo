"""Bilinear, trilinear and quadrilinear backbone fits following De Luca et al. (2013).

The fit splits a pushover curve at its peak. The elastic stiffness is the
secant to a small fraction of the peak force; a hardening line through the
peak is fitted to the rising branch; and a softening line from the peak,
optionally followed by a horizontal residual plateau, is fitted to the
falling branch. Each line is chosen to minimise the area between it and the
data. In ``"auto"`` mode the plateau is kept only when it clearly improves
the fit and is long and high enough to be meaningful. ``"bilinear-deluca"``
is the elastic-perfectly-plastic shape, with nothing optimised.

Every search runs only over parameter values that give a valid shape, so no
penalty values are needed and the optimisers never stall on a flat penalty
region.
"""

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from fitpo._common import (
    _REL_XTOL,
    FitResult,
    FloatArr,
    OriginMode,
    _as_curve,
    _check_finite,
    _finish,
    _grid_then_brent,
    _interval_area_error,
)

type DelucaMode = Literal["auto", "bilinear-deluca", "trilinear-deluca", "quadrilinear-deluca"]
_DELUCA_MODES: tuple[str, ...] = (
    "auto",
    "bilinear-deluca",
    "trilinear-deluca",
    "quadrilinear-deluca",
)

# Interior points of the coarse grid searches. Enough to land in the right
# basin on curves with several local minima, while keeping a fit well under a
# second.
_GRID_POINTS = 100

_NO_HARDENING_WARNING = (
    "trilinear/quadrilinear need a hardening branch, but the yield strength is equal "
    'to the maximum strength. Use mode="bilinear-deluca" or fit_cbf_cusum_backbone.'
)


def fit_deluca_backbone(
    pushover: npt.ArrayLike,
    *,
    mode: DelucaMode = "auto",
    elastic_ref_fraction: float = 0.10,
    plateau_min_improvement: float = 0.05,
    plateau_min_force_fraction: float = 0.10,
    plateau_min_length_fraction: float = 0.05,
    origin: OriginMode = "strict",
) -> FitResult:
    r"""Fit a bilinear, trilinear or quadrilinear backbone to a pushover curve.

    Parameters
    ----------
    pushover
        ``(N, 2)`` curve: column 0 displacement (never decreasing), column 1
        force. Any consistent units work; the examples use mm
        and N.
    mode
        ``"auto"`` (default) fits trilinear and quadrilinear and keeps the
        plateau only if the plateau-acceptance rule below holds.
        ``"bilinear-deluca"``, ``"trilinear-deluca"`` and
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
        ``fitted_po`` rows per mode:

        - ``"bilinear-deluca"``: origin, yield ``(F_max / k_e, F_max)``,
          end ``(d_max, F_max)``. ``objective`` is over ``[0, d_max]``.
        - ``"trilinear-deluca"``: origin, yield, peak, end.
        - ``"quadrilinear-deluca"``: origin, yield, peak, plateau start, end.

        For the trilinear and quadrilinear modes ``objective`` is over
        ``[d_y, d_max]``. When no hardening line exists (see Warns), the
        peak row is absent.

    Warns
    -----
    UserWarning
        For the trilinear and quadrilinear modes (and ``"auto"``) when no
        hardening line exists: the secant from the origin to the peak is at
        least as stiff as the elastic stiffness ``k_e``. The yield strength
        is then set equal to the maximum strength, keeping the stiffness
        ``k_e``, so the yield point is ``(F_max / k_e, F_max)`` with no
        hardening segment; the softening line (and plateau) start there.
        Also warns if any optimiser does not report convergence.

    Raises
    ------
    ValueError
        On invalid input, if the peak is at the last point (trilinear and
        quadrilinear modes), if ``F_max / k_e >= d_max`` where that point is
        needed, or if the fitted shape fails the validity checks.

    Notes
    -----
    The method follows De Luca et al. (2013). The elastic stiffness
    :math:`k_e` is the secant to ``elastic_ref_fraction`` of the peak force.
    Each later segment is chosen to minimise the area between it and the
    data,

    .. math::

        \int \left| F(d) - \hat F(d) \right| \, \mathrm{d}d ,

    where :math:`F` is the curve and :math:`\hat F` the fit.

    Examples
    --------
    >>> import numpy as np
    >>> import fitpo
    >>> d = np.linspace(0.0, 120.0, 241)
    >>> f = np.interp(d, [0, 10, 40, 80, 120], [0, 1000, 1200, 600, 600])
    >>> curve = np.column_stack([d, f])
    >>> res = fitpo.fit_deluca_backbone(curve)
    >>> res.mode
    'quadrilinear-deluca'
    >>> print(np.round(res.fitted_po, 1))
    [[   0.    0.]
     [  10. 1000.]
     [  40. 1200.]
     [  80.  600.]
     [ 120.  600.]]

    References
    ----------
    De Luca, F., Vamvatsikos, D. and Iervolino, I. (2013). Near-optimal
    piecewise linear fits of static pushover capacity curves for equivalent
    SDOF analysis. *Earthquake Engineering & Structural Dynamics*, 42(4),
    523-543. https://doi.org/10.1002/eqe.2225
    """
    if mode not in _DELUCA_MODES:
        raise ValueError(f"mode must be one of {_DELUCA_MODES}; got {mode!r}")
    d, f = _as_curve(pushover, 3, origin=origin)

    i_peak = int(np.argmax(f))
    d_peak = float(d[i_peak])
    f_peak = float(f[i_peak])
    if f_peak <= 0.0:
        raise ValueError("Peak force must be positive")
    k_e = elastic_slope(d[: i_peak + 1], f[: i_peak + 1], f_peak, ref_fraction=elastic_ref_fraction)
    d_max = float(d[-1])

    if mode == "bilinear-deluca":
        # Elastic-perfectly-plastic: nothing to optimise, so it always "converges".
        d_y = _yield_at_peak_force(f_peak, k_e, d_max)
        bp = np.asarray([[0.0, 0.0], [d_y, f_peak], [d_max, f_peak]], dtype=np.float64)
        obj = _interval_area_error(d, f, bp, 0.0, d_max)
        return _finish(bp, mode, obj, True, "fit_deluca_backbone")

    if i_peak >= len(f) - 1:
        raise ValueError("No descending branch: peak is at the last data point")

    # Success flags of every optimiser call, so `converged` covers all of them.
    ok: list[bool] = []

    # The rising branch. A hardening line through the peak meets the elastic
    # line before the peak only if the secant to the peak is less stiff than
    # k_e. Otherwise there is no valid hardening line, and the yield point
    # becomes (F_max / k_e, F_max): the limit of a zero-slope hardening line.
    if f_peak / d_peak >= k_e:
        warnings.warn(_NO_HARDENING_WARNING, UserWarning, stacklevel=2)
        d_s = _yield_at_peak_force(f_peak, k_e, d_max)
        head = [[0.0, 0.0], [d_s, f_peak]]
    else:
        d_y, f_y = _fit_hardening(d, f, d_peak, f_peak, k_e, ok)
        d_s = d_peak
        head = [[0.0, 0.0], [d_y, f_y], [d_peak, f_peak]]

    # The falling branch starts at (d_s, f_peak): the peak, or the yield
    # point when there is no hardening line.
    def tri() -> tuple[FloatArr, float]:
        k_s, err = _fit_softening(d, f, d_s, f_peak, d_max, ok)
        tail = [[d_max, f_peak + k_s * (d_max - d_s)]]
        return np.asarray(head + tail, dtype=np.float64), err

    def quad() -> tuple[FloatArr, float]:
        d_b, f_p, err = _fit_softening_plateau(d, f, d_s, f_peak, d_max, ok)
        tail = [[d_b, f_p], [d_max, f_p]]
        return np.asarray(head + tail, dtype=np.float64), err

    if mode == "trilinear-deluca":
        bp, _ = tri()
    elif mode == "quadrilinear-deluca":
        bp, _ = quad()
    else:
        # Auto: compare the two falling-branch fits by their area error over
        # the falling branch only (the rising branch is the same for both).
        bp3, a3 = tri()
        bp4, a4 = quad()
        d_b, f_p = float(bp4[-2, 0]), float(bp4[-2, 1])
        # Plateau-acceptance rule: a plateau adds two parameters, so keep it
        # only if it pays for them with a clear improvement, and if it is
        # neither a near-zero force level nor a short stub at the end.
        accept_plateau = (
            a4 < (1.0 - plateau_min_improvement) * a3
            and f_p >= plateau_min_force_fraction * f_peak
            and d_max - d_b >= plateau_min_length_fraction * (d_max - d_s)
        )
        bp = bp4 if accept_plateau else bp3
    fitted_mode = "quadrilinear-deluca" if len(bp) == len(head) + 2 else "trilinear-deluca"

    obj = _interval_area_error(d, f, bp, float(bp[1, 0]), d_max)
    return _finish(bp, fitted_mode, obj, all(ok), "fit_deluca_backbone")


def elastic_slope(
    d_asc: npt.ArrayLike,
    f_asc: npt.ArrayLike,
    f_peak: float,
    *,
    ref_fraction: float = 0.10,
) -> float:
    """Secant stiffness from the origin to ``ref_fraction * f_peak`` on the rising branch.

    The reference point is where the rising branch first reaches
    ``ref_fraction * f_peak``, interpolated linearly between the two samples
    either side of that crossing. Walking to the first crossing keeps the
    result well defined when the force is not monotone (e.g. small dips
    before the peak). If the first sample is already at or above
    ``ref_fraction * f_peak``, the crossing lies before the data and a
    ``ValueError`` is raised.

    Parameters
    ----------
    d_asc, f_asc
        Displacement and force of the rising branch, from the first point up
        to and including the peak. Any consistent units work; the examples use mm
        and N.
    f_peak
        Peak force of the curve.
    ref_fraction
        Fraction of ``f_peak`` that sets the reference point; default 0.10.

    Returns
    -------
    float
        The secant stiffness ``ref_fraction * f_peak / d_ref``, in force per
        unit displacement.

    Raises
    ------
    ValueError
        If the arrays are not 1-D and of equal, non-zero length, hold NaN or
        inf, never reach ``ref_fraction * f_peak``, already reach it at the
        first sample, or if the reference displacement is not positive.

    Examples
    --------
    The reference force is ``0.10 * 250 = 25``, reached at ``d = 0.5``:

    >>> import fitpo
    >>> d = [0.0, 1.0, 2.0, 3.0, 4.0]
    >>> f = [0.0, 50.0, 120.0, 200.0, 250.0]
    >>> fitpo.elastic_slope(d, f, 250.0)
    50.0

    References
    ----------
    De Luca, F., Vamvatsikos, D. and Iervolino, I. (2013). Near-optimal
    piecewise linear fits of static pushover capacity curves for equivalent
    SDOF analysis. *Earthquake Engineering & Structural Dynamics*, 42(4),
    523-543. https://doi.org/10.1002/eqe.2225
    """
    d_arr = np.asarray(d_asc, dtype=np.float64)
    f_arr = np.asarray(f_asc, dtype=np.float64)
    if d_arr.ndim != 1 or d_arr.shape != f_arr.shape or d_arr.size == 0:
        raise ValueError("d_asc and f_asc must be 1-D arrays of the same, non-zero length")
    _check_finite(np.column_stack([d_arr, f_arr]), "d_asc/f_asc")
    if not np.isfinite(f_peak):
        raise ValueError("f_peak contains NaN or inf values")
    f_ref = ref_fraction * f_peak

    above = np.flatnonzero(f_arr >= f_ref)
    if above.size == 0:
        raise ValueError("The rising branch never reaches ref_fraction * f_peak")
    i = int(above[0])
    if i == 0:
        # The crossing lies before the first sample, so it can't be located
        # from the data and the stiffness is undefined.
        raise ValueError(
            "The first sample already reaches ref_fraction * f_peak, so the elastic "
            "reference point lies before the data; check the curve starts at its "
            "unloaded point, or use origin='correct_dispforc'"
        )
    if f_arr[i] == f_ref:
        # Exactly on a sample: use it directly.
        d_ref = float(d_arr[i])
    else:
        # Same operation order as np.interp, so a monotone rising branch
        # gives bit-identical results to interpolating with force as x-axis.
        slope = (d_arr[i] - d_arr[i - 1]) / (f_arr[i] - f_arr[i - 1])
        d_ref = float(slope * (f_ref - f_arr[i - 1]) + d_arr[i - 1])
    if d_ref <= 0.0:
        raise ValueError("Elastic reference displacement <= 0")
    return f_ref / d_ref


def _yield_at_peak_force(f_peak: float, k_e: float, d_max: float) -> float:
    """Displacement where the elastic line reaches ``f_peak``; must lie inside the data."""
    d_y = f_peak / k_e
    if d_y >= d_max:
        raise ValueError(
            f"The elastic line reaches the maximum force at d = F_max / k_e = {d_y:.6g}, "
            f"at or beyond the last data point d_max = {d_max:.6g}"
        )
    return d_y


def _fit_hardening(
    d: FloatArr,
    f: FloatArr,
    d_peak: float,
    f_peak: float,
    k_e: float,
    ok: list[bool],
) -> tuple[float, float]:
    """Fit the hardening line through the peak; return the yield point ``(d_y, f_y)``.

    The yield point is where the hardening line meets the elastic line
    ``F = k_e d``, so only the hardening slope ``k_h`` is free. It is searched
    over ``[0, f_peak / d_peak)`` only: there ``0 < d_y < d_peak`` (the caller
    has checked ``f_peak / d_peak < k_e``), so every trial is a valid shape.
    The line is scored against the data over ``[d_y, d_peak]``.
    """

    def yield_point(k_h: float) -> tuple[float, float]:
        d_y = (f_peak - k_h * d_peak) / (k_e - k_h)
        return d_y, k_e * d_y

    def obj(k_h: float) -> float:
        d_y, f_y = yield_point(k_h)
        bp = np.asarray([[d_y, f_y], [d_peak, f_peak]], dtype=np.float64)
        return _interval_area_error(d, f, bp, d_y, d_peak)

    # k_h = 0 is a valid shape (a flat hardening line), so it is tested too.
    k_h, err = _grid_then_brent(obj, 0.0, f_peak / d_peak, _GRID_POINTS, ok)
    if obj(0.0) < err:
        k_h = 0.0
    return yield_point(k_h)


def _fit_softening(
    d: FloatArr,
    f: FloatArr,
    d_s: float,
    f_s: float,
    d_max: float,
    ok: list[bool],
) -> tuple[float, float]:
    """Fit one softening line from ``(d_s, f_s)`` to ``d_max``; return ``(k_s, error)``.

    The area error is convex in the slope, so a bounded Brent search finds
    the optimum. The steepest slope searched reaches zero force ten times
    faster than over the whole falling branch.
    """
    k_s_min = -f_s / (d_max - d_s) * 10.0

    def obj(k_s: float) -> float:
        bp = np.asarray([[d_s, f_s], [d_max, f_s + k_s * (d_max - d_s)]], dtype=np.float64)
        return _interval_area_error(d, f, bp, d_s, d_max)

    res = minimize_scalar(
        obj, bounds=(k_s_min, 0.0), method="bounded", options={"xatol": _REL_XTOL * -k_s_min}
    )
    ok.append(bool(res.success))
    return float(res.x), float(res.fun)


def _fit_softening_plateau(
    d: FloatArr,
    f: FloatArr,
    d_s: float,
    f_s: float,
    d_max: float,
    ok: list[bool],
) -> tuple[float, float, float]:
    """Fit a softening line + horizontal plateau; return ``(d_b, f_p, error)``.

    The line runs from ``(d_s, f_s)`` to the plateau start ``(d_b, f_p)``,
    then the plateau stays at ``f_p`` to ``d_max``. It is searched as an
    outer search over ``d_b`` in ``(d_s, d_max)`` with an inner search over
    ``f_p`` in ``(0, f_s)``, so every trial is a valid, falling shape. For a
    fixed ``d_b`` the area error is convex in ``f_p``, so a bounded Brent
    search solves the inner problem; the outer one can have several local
    minima, so it uses a coarse grid first.
    """
    inner_ok: list[bool] = []
    # The data on [d_s, d_max], with both ends interpolated; fixed for the whole
    # search, so it is built once.
    inside = (d > d_s) & (d < d_max)
    d_rng = np.concatenate(([d_s], d[inside], [d_max]))
    f_rng = np.interp(d_rng, d, f)

    def best_plateau(d_b: float) -> tuple[float, float]:
        # For a fixed d_b the grid (the data plus the knot d_b) and the shape
        # of the fit are fixed: the fit is f_s * (1 - t) + f_p * t, with t
        # rising from 0 at d_s to 1 at d_b and staying 1 on the plateau. Only
        # f_p changes in the inner search, so each trial is a few array
        # operations. This is the same area error as _interval_area_error.
        k = int(np.searchsorted(d_rng, d_b))
        grid = np.insert(d_rng, k, d_b)
        f_grid = np.insert(f_rng, k, float(np.interp(d_b, d, f)))
        t = np.minimum((grid - d_s) / (d_b - d_s), 1.0)
        base = f_grid - f_s * (1.0 - t)

        def obj(f_p: float) -> float:
            return float(np.trapezoid(np.abs(base - f_p * t), grid))

        res = minimize_scalar(
            obj, bounds=(0.0, f_s), method="bounded", options={"xatol": _REL_XTOL * f_s}
        )
        inner_ok.append(bool(res.success))
        return float(res.x), float(res.fun)

    d_b, err = _grid_then_brent(lambda x: best_plateau(x)[1], d_s, d_max, _GRID_POINTS, ok)
    f_p, err = best_plateau(d_b)
    ok.append(all(inner_ok))
    return d_b, f_p, err
