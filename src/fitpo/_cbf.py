"""Backbone fits for concentrically braced frame (CBF) pushover curves.

A CBF pushover curve typically rises elastically, softens as the
compression braces buckle, then drops suddenly when a brace fractures and
settles on a lower residual branch. The De Luca fits cannot represent that
sudden drop, so this module offers two fits built around it:

- :func:`fit_cbf_cusum_backbone` finds the fracture drop automatically with
  a CUSUM change detector on the slope of the curve, then fits the
  post-buckling and residual lines around it.
- :func:`fit_cbf_knots_backbone` places the two knots of the drop either at
  force fractions of the peak or at given displacements, and fits all knot
  forces together by area minimisation.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, minimize_scalar

from fitpo._common import FitResult, FloatArr, _area_error, _as_curve, _interval_area_error
from fitpo._deluca import elastic_slope

type CbfCusumMode = Literal["quadrilinear-cbf-cusum"]
type CbfKnotsMode = Literal["quadrilinear-cbf-knots", "trilinear-cbf-knots"]


def fit_cbf_cusum_backbone(
    pushover: npt.ArrayLike,
    *,
    cusum_threshold: float = 4.0,
    cusum_drift: float = 0.5,
    cusum_search_start_fraction: float = 0.25,
    elastic_ref_fraction: float = 0.10,
) -> FitResult:
    """Fit a 4-segment backbone to a CBF pushover curve, locating fracture by CUSUM.

    Parameters
    ----------
    pushover
        ``(N, 2)`` curve: column 0 displacement (increasing), column 1 force.
    cusum_threshold
        CUSUM trigger threshold on the standardised slope of the curve.
    cusum_drift
        CUSUM drift parameter on the standardised slope of the curve.
    cusum_search_start_fraction
        CUSUM starts searching at the first sample with
        ``d >= d_max * cusum_search_start_fraction``. Raise it for curves
        with small force drops before the fracture.
    elastic_ref_fraction
        The elastic stiffness is the secant to this fraction of the peak
        force on the rising branch.

    Returns
    -------
    FitResult
        ``fitted_po`` has 5 rows:
        ``[[0, 0], [d_y, f_y], [d_joint, f_joint], [d_r2, f_r2], [d_max, f_end]]``.
        Rows 2 and 3 lie on the fracture-drop line found by CUSUM; the other
        knots are chosen so segments 2 and 4 each minimise their area error
        against the data. ``mode`` is ``"quadrilinear-cbf-cusum"``;
        ``objective`` is the area error over ``[d_y, d_max]``.
    """
    d, f = _as_curve(pushover, 5)
    n = len(d)

    i_peak = int(np.argmax(f))
    f_peak = float(f[i_peak])
    if f_peak <= 0.0:
        raise ValueError("Peak force must be positive")
    if i_peak >= n - 1:
        raise ValueError("No descending branch: peak is at the last data point")

    k_e = elastic_slope(d[: i_peak + 1], f[: i_peak + 1], f_peak, ref_fraction=elastic_ref_fraction)

    d_max = float(d[-1])
    search_start_d = d_max * cusum_search_start_fraction
    i_f, i_r = _detect_fracture_region(
        d,
        f,
        search_start_d,
        cusum_threshold,
        cusum_drift,
    )

    d_f, f_f = float(d[i_f]), float(f[i_f])
    d_r, f_r = float(d[i_r]), float(f[i_r])

    # Success flags of every optimiser call, so `converged` covers all of them.
    ok: list[bool] = []
    _k_pb, d_y, f_y, d_joint, f_joint = _fit_post_buckling(
        d, f, d_f, f_f, d_r, f_r, k_e, f_peak, ok
    )
    _k_r, d_r2, f_r2, f_end = _fit_residual(d, f, d_r, f_r, d_max, ok)

    bp = np.asarray(
        [[0.0, 0.0], [d_y, f_y], [d_joint, f_joint], [d_r2, f_r2], [d_max, f_end]],
        dtype=np.float64,
    )
    return FitResult(
        bp, "quadrilinear-cbf-cusum", _interval_area_error(d, f, bp, d_y, d_max), all(ok)
    )


def fit_cbf_knots_backbone(
    pushover: npt.ArrayLike,
    *,
    joint_force_fraction: float = 0.80,
    residual_force_fraction: float = 0.05,
    residual_plateau: bool = True,
    d_f: float | None = None,
    d_r: float | None = None,
    elastic_ref_fraction: float = 0.10,
) -> FitResult:
    """Fit a CBF backbone with fixed knot displacements by joint area minimisation.

    Segments: elastic (0 -> y), post-buckling decline (y -> f), fracture
    drop (f -> r), and a horizontal residual plateau (r -> max). ``f_y``,
    ``f_f`` and ``f_r`` are found together by minimising the area between
    segments 2-4 and the data over ``[d_y, d_max]``.

    The two interior knot *displacements* can be set either way:

    - **By force fraction** (default): ``d_f`` and ``d_r`` are snapped to
      data points near the descending crossings of
      ``joint_force_fraction * f_peak`` and ``residual_force_fraction *
      f_peak``.
    - **By displacement**: pass ``d_f`` and ``d_r`` directly and the
      snapping is skipped, along with the force fractions.

    The distinction matters when the knots are being *searched over*.
    Snapping resolves a fraction to "the first data point at or below
    ``fraction * f_peak``", which is not monotone in the sampling of the
    curve: inserting extra points along a segment -- which leaves the
    piecewise-linear curve mathematically unchanged -- can change which
    point a given fraction resolves to, and so can make the best
    achievable fit *worse*. Addressing the knots by displacement removes
    that indirection, so a finer candidate set can only improve the fit.

    Parameters
    ----------
    pushover
        ``(N, 2)`` curve: column 0 displacement (increasing), column 1 force.
    joint_force_fraction
        Target for ``d_f`` is ``fraction * f_peak``; default 0.80.
        Ignored when ``d_f``/``d_r`` are given.
    residual_force_fraction
        Target for ``d_r`` is ``fraction * f_peak``; default 0.05.
        Must satisfy ``0 < residual_force_fraction < joint_force_fraction < 1``.
        Ignored when ``d_f``/``d_r`` are given.
    residual_plateau
        If ``True`` (default), the curve ends on a horizontal residual
        plateau. If ``False``, there is no plateau: the fracture drop
        continues with its own slope down to zero force.
    d_f, d_r
        Interior knot displacements, in the units of column 0. Give both
        or neither; they must satisfy ``0 < d_f < d_r <= d_max``. Giving
        them bypasses the force-fraction snapping entirely.
    elastic_ref_fraction
        The elastic stiffness is the secant to this fraction of the peak
        force on the rising branch.

    Returns
    -------
    FitResult
        With ``residual_plateau=True``: ``mode`` is
        ``"quadrilinear-cbf-knots"`` and ``fitted_po`` is
        ``[[0, 0], [d_y, f_y], [d_f, f_f], [d_r, f_r], [d_max, f_r]]``.
        With ``residual_plateau=False``: ``mode`` is
        ``"trilinear-cbf-knots"`` and ``fitted_po`` is
        ``[[0, 0], [d_y, f_y], [d_f, f_f], [d_zero, 0], [d_zero, 0]]``.
        ``objective`` is the area error over ``[d_y, d_max]``.

    Notes
    -----
    ``d_y`` is not an independent parameter: it is ``f_y / k_e``, with
    ``k_e`` the secant stiffness at ``elastic_ref_fraction`` of the peak
    force on the rising branch. Setting ``d_f``/``d_r`` therefore gives
    no control over ``d_y``.
    """
    if (d_f is None) != (d_r is None):
        raise ValueError("Give both d_f and d_r, or neither")

    d, f = _as_curve(pushover, 5)
    n = len(d)

    i_peak = int(np.argmax(f))
    f_peak = float(f[i_peak])
    if f_peak <= 0.0:
        raise ValueError("Peak force must be positive")
    if i_peak >= n - 1:
        raise ValueError("No descending branch: peak is at the last data point")

    k_e = elastic_slope(d[: i_peak + 1], f[: i_peak + 1], f_peak, ref_fraction=elastic_ref_fraction)
    d_max = float(d[-1])

    if d_f is not None and d_r is not None:
        # Knots given directly: no snapping, so they need not be data points.
        # Seed the inner solver from the curve at those displacements, which is
        # what the snapping path passes from the points it landed on.
        d_f_knot = float(d_f)
        d_r_knot = float(d_r)
        if not 0.0 < d_f_knot < d_r_knot <= d_max:
            raise ValueError(
                f"Require 0 < d_f < d_r <= d_max; got d_f={d_f_knot:.6g}, "
                f"d_r={d_r_knot:.6g}, d_max={d_max:.6g}"
            )
        f_f_init = float(np.interp(d_f_knot, d, f))
        f_r_init = float(np.interp(d_r_knot, d, f))
    else:
        # The fractions are only meaningful on the snapping path.
        if not 0.0 < residual_force_fraction < joint_force_fraction < 1.0:
            raise ValueError("Require 0 < residual_force_fraction < joint_force_fraction < 1")
        i_f = _snap_descending_crossing(f, i_peak, joint_force_fraction * f_peak)
        if i_f is None:
            raise ValueError("No descending crossing of joint_force_fraction * f_peak")
        i_r = _snap_descending_crossing(f, i_f, residual_force_fraction * f_peak)
        if i_r is None:
            i_r = n - 1
        if i_f >= i_r:
            raise ValueError("d_f must precede d_r; choose different fractions")

        d_f_knot = float(d[i_f])
        d_r_knot = float(d[i_r])
        f_f_init = float(f[i_f])
        f_r_init = float(f[i_r])

    bp, success = _fit_tetra_free(
        d,
        f,
        k_e,
        d_f_knot,
        d_r_knot,
        d_max,
        f_peak,
        f_f_init=f_f_init,
        f_r_init=f_r_init,
        collapse_residual=not residual_plateau,
    )
    mode = "quadrilinear-cbf-knots" if residual_plateau else "trilinear-cbf-knots"
    d_y = float(bp[1, 0])
    return FitResult(bp, mode, _interval_area_error(d, f, bp, d_y, d_max), success)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _snap_descending_crossing(
    f: FloatArr, i_start: int, target: float,
) -> int | None:
    """Snap to the data point nearest the descending crossing of ``target``.

    Walks from ``i_start + 1`` until the first index ``i`` with
    ``f[i] <= target``. Returns whichever of ``{i - 1, i}`` has ``f``
    closer to ``target``. Returns ``None`` if no descending crossing
    exists past ``i_start``.
    """
    after = f[i_start + 1 :]
    below = np.flatnonzero(after <= target)
    if below.size == 0:
        return None
    i_cross = int(below[0]) + i_start + 1
    return i_cross - 1 if abs(f[i_cross - 1] - target) <= abs(f[i_cross] - target) else i_cross


def _detect_fracture_region(
    d: FloatArr,
    f: FloatArr,
    search_start_d: float,
    cusum_threshold: float,
    cusum_drift: float,
) -> tuple[int, int]:
    """Locate fracture endpoints via CUSUM on standardised velocity.

    Returns global indices (i_f, i_r) bounding the fracture event.
    """
    matches = np.flatnonzero(d >= search_start_d)
    if matches.size == 0:
        raise ValueError(
            f"CUSUM search start {search_start_d:.4g} exceeds d_max = {float(d[-1]):.4g}"
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


def _fit_kpb_at(
    d: FloatArr,
    f: FloatArr,
    d_joint: float,
    f_joint: float,
    k_e: float,
    ok: list[bool],
) -> tuple[float, float, float, float]:
    """Inner: 1-D Brent over k_pb, post-buckling line pinned at (d_joint, f_joint).

    Returns (k_pb, d_y, f_y, area_err).
    """
    k_pb_min = -abs(f_joint) / max(d_joint, 1e-9) * 10.0
    # Geometric upper bound: d_y > 0 requires k_pb < f_joint / d_joint.
    k_pb_max = min(k_e * 0.999, f_joint / max(d_joint, 1e-9) * 0.999)

    def obj(k_pb: float) -> float:
        denom = k_e - k_pb
        if denom <= 0.0:
            return 1e18
        d_y = (f_joint - k_pb * d_joint) / denom
        if d_y <= 0.0 or d_y >= d_joint:
            return 1e18
        mask = (d >= d_y) & (d <= d_joint)
        if not bool(mask.any()):
            return 1e18
        return _area_error(d[mask], f[mask], f_joint + k_pb * (d[mask] - d_joint))

    if k_pb_max <= k_pb_min:
        k_pb = 0.0
        d_y = f_joint / max(k_e, 1e-9)
        return k_pb, d_y, k_e * d_y, _segment_area(d, f, d_y, d_joint, f_joint, k_pb)

    res = minimize_scalar(obj, bounds=(k_pb_min, k_pb_max), method="bounded")
    ok.append(bool(res.success))
    k_pb = float(res.x)
    denom = k_e - k_pb
    if denom <= 0.0:
        k_pb = 0.0
        denom = k_e
    d_y = (f_joint - k_pb * d_joint) / denom
    if d_y <= 0.0:
        d_y = f_joint / max(k_e, 1e-9)
        k_pb = 0.0
    f_y = k_e * d_y
    return k_pb, d_y, f_y, _segment_area(d, f, d_y, d_joint, f_joint, k_pb)


def _segment_area(
    d: FloatArr,
    f: FloatArr,
    d_y: float,
    d_joint: float,
    f_joint: float,
    k_pb: float,
) -> float:
    mask = (d >= d_y) & (d <= d_joint)
    if not bool(mask.any()):
        return 1e18
    return _area_error(d[mask], f[mask], f_joint + k_pb * (d[mask] - d_joint))


def _fit_post_buckling(
    d: FloatArr,
    f: FloatArr,
    d_f: float,
    f_f: float,
    d_r: float,
    f_r: float,
    k_e: float,
    f_peak: float,
    ok: list[bool],
) -> tuple[float, float, float, float, float]:
    """Free f_joint fitting; d_joint = d_f fixed.

    Outer Brent over f_joint in (f_r, 1.5*f_peak); minimises combined
    segment-2 + segment-3 absolute area error.

    Returns (k_pb, d_y, f_y, d_joint=d_f, f_joint).
    """
    f_joint_max = 1.5 * f_peak
    f_joint_min = f_r + 1e-6

    if f_joint_min >= f_joint_max:
        k_pb, d_y, f_y, _ = _fit_kpb_at(d, f, d_f, f_r, k_e, ok)
        return k_pb, d_y, f_y, d_f, f_r

    def outer_obj(f_joint: float) -> float:
        k3 = (f_r - f_joint) / max(d_r - d_f, 1e-9)
        k_pb, _d_y, _f_y, area2 = _fit_kpb_at(d, f, d_f, f_joint, k_e, ok)
        mask = (d >= d_f) & (d <= d_r)
        if not bool(mask.any()):
            return area2
        return area2 + _area_error(d[mask], f[mask], f_joint + k3 * (d[mask] - d_f))

    res = minimize_scalar(outer_obj, bounds=(f_joint_min, f_joint_max), method="bounded")
    ok.append(bool(res.success))
    f_joint = float(res.x)
    k_pb, d_y, f_y, _ = _fit_kpb_at(d, f, d_f, f_joint, k_e, ok)
    return k_pb, d_y, f_y, d_f, f_joint


def _fit_kr_at(
    d: FloatArr,
    f: FloatArr,
    d_r2: float,
    f_r2: float,
    d_max: float,
    ok: list[bool],
) -> tuple[float, float, float]:
    """Inner: 1-D Brent over k_r ≤ 0, residual line pinned at (d_r2, f_r2).

    Returns (k_r, f_end, area_err).
    """
    span = d_max - d_r2
    if span < 1e-9:
        return 0.0, f_r2, 0.0

    k_r_min = -f_r2 / span * 2.0

    def obj(k_r: float) -> float:
        mask = d >= d_r2
        if not bool(mask.any()):
            return 1e18
        return _area_error(d[mask], f[mask], f_r2 + k_r * (d[mask] - d_r2))

    res = minimize_scalar(obj, bounds=(k_r_min, 0.0), method="bounded")
    ok.append(bool(res.success))
    k_r = float(res.x)
    return k_r, f_r2 + k_r * span, float(res.fun)


def _fit_residual(
    d: FloatArr,
    f: FloatArr,
    d_r: float,
    f_r: float,
    d_max: float,
    ok: list[bool],
) -> tuple[float, float, float, float]:
    """Vertical sliding: d_r2 = d_r fixed, f_r2 free in (1e-6, f_r).

    Outer Brent over f_r2; inner k_r <= 0 from _fit_kr_at.
    Returns (k_r, d_r2=d_r, f_r2, f_end).
    """
    if f_r <= 1e-6 or d_max <= d_r + 1e-9:
        return 0.0, d_r, f_r, f_r

    def outer_obj(f_r2: float) -> float:
        return _fit_kr_at(d, f, d_r, f_r2, d_max, ok)[2]

    res = minimize_scalar(outer_obj, bounds=(1e-6, 2 * f_r), method="bounded")
    ok.append(bool(res.success))
    f_r2 = float(res.x)
    k_r, f_end, _ = _fit_kr_at(d, f, d_r, f_r2, d_max, ok)
    return k_r, d_r, f_r2, f_end


def _fit_tetra_free(
    d: FloatArr,
    f: FloatArr,
    k_e: float,
    d_f: float,
    d_r: float,
    d_max: float,
    f_peak: float,
    *,
    f_f_init: float,
    f_r_init: float,
    collapse_residual: bool,
) -> tuple[FloatArr, bool]:
    """Joint L-BFGS-B over (f_y, f_f, f_r); return 5x2 breakpoints and success."""
    upper = 1.5 * f_peak
    penalty = 1e3 * f_peak * max(d_max, 1.0)

    def piecewise_fit(d_grid: FloatArr, d_y: float, f_y: float,
                      f_f: float, f_r: float) -> FloatArr:
        xs = np.asarray([d_y, d_f, d_r, d_max], dtype=np.float64)
        ys = np.asarray([f_y, f_f, f_r, f_r], dtype=np.float64)
        return np.interp(d_grid, xs, ys)

    def piecewise_fit_collapse(d_grid: FloatArr, d_y: float, f_y: float,
                               f_f: float, f_r: float) -> tuple[FloatArr, float]:
        k3 = (f_r - f_f) / (d_r - d_f)
        d_zero = d_f - f_f / k3  # k3 < 0 guaranteed by caller
        d_end = max(d_max, d_zero)
        xs = np.asarray([d_y, d_f, d_zero, d_end], dtype=np.float64)
        ys = np.asarray([f_y, f_f, 0.0, 0.0], dtype=np.float64)
        return np.interp(d_grid, xs, ys), d_zero

    def objective(x: FloatArr) -> float:
        f_y, f_f, f_r = float(x[0]), float(x[1]), float(x[2])
        d_y = f_y / k_e
        pen = 0.0
        if d_y <= 0.0 or d_y >= d_f:
            pen += penalty
        if f_r > f_f:
            pen += penalty * (f_r - f_f) / max(f_peak, 1e-9)
        if collapse_residual:
            k3 = (f_r - f_f) / (d_r - d_f)
            if k3 >= -1e-12:
                return penalty
            d_y_eff = max(d_y, 1e-6)
            d_zero = d_f - f_f / k3
            d_end = max(d_max, d_zero)
            d_data = d[d >= d_y_eff]
            grid = np.unique(np.concatenate(
                ([d_y_eff], d_data, [d_f, d_zero, d_end])
            ))
            f_data = np.interp(grid, d, f)
            f_fit, _ = piecewise_fit_collapse(grid, d_y_eff, f_y, f_f, f_r)
            return _area_error(grid, f_data, f_fit) + pen
        d_y_eff = max(d_y, 1e-6)
        d_data = d[d >= d_y_eff]
        grid = np.unique(np.concatenate(([d_y_eff], d_data, [d_f, d_r, d_max])))
        f_data = np.interp(grid, d, f)
        f_fit = piecewise_fit(grid, d_y_eff, f_y, f_f, f_r)
        return _area_error(grid, f_data, f_fit) + pen

    x0 = np.asarray(
        [min(f_peak, upper), min(max(f_f_init, 1e-6), upper),
         min(max(f_r_init, 1e-6), upper)],
        dtype=np.float64,
    )
    bounds = [(1e-6, upper)] * 3
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    f_y, f_f, f_r = float(res.x[0]), float(res.x[1]), float(res.x[2])
    d_y = f_y / k_e

    if collapse_residual:
        k3 = (f_r - f_f) / (d_r - d_f)
        if k3 >= 0.0:
            k3 = -1e-12
        d_zero = d_f - f_f / k3
        bp = np.asarray(
            [[0.0, 0.0], [d_y, f_y], [d_f, f_f], [d_zero, 0.0], [d_zero, 0.0]],
            dtype=np.float64,
        )
    else:
        bp = np.asarray(
            [[0.0, 0.0], [d_y, f_y], [d_f, f_f], [d_r, f_r], [d_max, f_r]],
            dtype=np.float64,
        )
    return bp, bool(res.success)
