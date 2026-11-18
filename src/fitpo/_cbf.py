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

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog, minimize_scalar

from fitpo._common import (
    FitResult,
    FloatArr,
    OriginMode,
    _area_error,
    _as_curve,
    _finish,
    _grid_then_brent,
    _interval_area_error,
)
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
    origin: OriginMode = "strict",
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
        ``fitted_po`` has 5 rows:
        ``[[0, 0], [d_y, f_y], [d_joint, f_joint], [d_r2, f_r2], [d_max, f_end]]``.
        Rows 2 and 3 lie on the fracture-drop line found by CUSUM; the other
        knots are chosen so segments 2 and 4 each minimise their area error
        against the data. ``mode`` is ``"quadrilinear-cbf-cusum"``;
        ``objective`` is the area error over ``[d_y, d_max]``.
    """
    d, f = _as_curve(pushover, 5, origin=origin)
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
    obj = _interval_area_error(d, f, bp, d_y, d_max)
    return _finish(bp, "quadrilinear-cbf-cusum", obj, all(ok), "fit_cbf_cusum_backbone")


def fit_cbf_knots_backbone(
    pushover: npt.ArrayLike,
    *,
    joint_force_fraction: float = 0.80,
    residual_force_fraction: float = 0.05,
    residual_plateau: bool = True,
    d_f: float | None = None,
    d_r: float | None = None,
    elastic_ref_fraction: float = 0.10,
    origin: OriginMode = "strict",
) -> FitResult:
    """Fit a CBF backbone with fixed knot displacements by joint area minimisation.

    Segments: elastic (0 -> y), post-buckling decline (y -> f), then either
    a fracture drop (f -> r) and a horizontal residual plateau (r -> max), or
    a single drop to zero force. The knot forces are found together by
    minimising the area between the fit and the data over ``[d_y, d_max]``;
    nothing beyond the last data point is scored.

    The interior knot *displacements* can be set either way:

    - **By force fraction** (default): ``d_f`` (and ``d_r`` with the
      plateau) are snapped to data points near the descending crossings of
      ``joint_force_fraction * f_peak`` and ``residual_force_fraction *
      f_peak``.
    - **By displacement**: pass ``d_f`` (and ``d_r`` with the plateau)
      directly and the snapping is skipped, along with the force fractions.

    With force fractions, a meaningful quadrilinear fit needs the curve to
    cross both levels after the peak. If it never falls to
    ``joint_force_fraction * f_peak``, a ``ValueError`` is raised. If it
    never falls to ``residual_force_fraction * f_peak``, ``d_r`` becomes the
    last data point, the plateau has zero length and the fit has only three
    branches; a ``UserWarning`` suggests raising ``residual_force_fraction``.

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
        ``(N, 2)`` curve: column 0 displacement (never decreasing), column 1
        force.
    joint_force_fraction
        Target for ``d_f`` is ``fraction * f_peak``; default 0.80.
        Ignored when ``d_f`` is given.
    residual_force_fraction
        Target for ``d_r`` is ``fraction * f_peak``; default 0.05. Must
        satisfy ``0 < residual_force_fraction < joint_force_fraction < 1``.
        Ignored when ``d_r`` is given, and ignored entirely when
        ``residual_plateau=False``.
    residual_plateau
        If ``True`` (default), the curve ends on a horizontal residual
        plateau. If ``False``, there is no plateau and no ``d_r``: the drop
        from ``(d_f, f_f)`` continues with its own fitted slope down to zero
        force.
    d_f, d_r
        Interior knot displacements, in the units of column 0 and in the
        coordinates of the curve as passed in; with ``origin="correct_disp"``
        or ``"correct_dispforc"`` they are shifted by ``d[0]`` along with the
        curve. With the plateau, give both or neither; after any shift they
        must satisfy ``0 < d_f < d_r <= d_max``. With
        ``residual_plateau=False`` only ``d_f`` is used (``0 < d_f <
        d_max``) and ``d_r`` is ignored. Giving them bypasses the
        force-fraction snapping.
    elastic_ref_fraction
        The elastic stiffness is the secant to this fraction of the peak
        force on the rising branch.
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
        With ``residual_plateau=True``: ``mode`` is
        ``"quadrilinear-cbf-knots"`` and ``fitted_po`` is
        ``[[0, 0], [d_y, f_y], [d_f, f_f], [d_r, f_r], [d_max, f_r]]``; the
        last row is dropped when ``d_r == d_max``.
        With ``residual_plateau=False``: ``mode`` is
        ``"trilinear-cbf-knots"`` and ``fitted_po`` is
        ``[[0, 0], [d_y, f_y], [d_f, f_f], [d_zero, 0]]``, followed by
        ``[d_max, 0]`` when the drop reaches zero force before ``d_max``.
        ``objective`` is the area error over ``[d_y, d_max]``.

    Notes
    -----
    ``d_y`` is not an independent parameter: it is ``f_y / k_e``, with
    ``k_e`` the secant stiffness at ``elastic_ref_fraction`` of the peak
    force on the rising branch. Setting ``d_f``/``d_r`` therefore gives
    no control over ``d_y``.

    For a fixed ``f_y`` the fitted curve is linear in the other knot
    forces, so they are solved exactly (a linear program with the plateau, a
    weighted median for the drop). ``f_y``, and the zero-force point of the
    drop, are found by 1-D searches.
    """
    if residual_plateau and (d_f is None) != (d_r is None):
        raise ValueError("Give both d_f and d_r, or neither")

    # Explicit knots are given in the input curve's coordinates, so record the
    # shift before applying it and move the knots with the curve.
    d, f = _as_curve(pushover, 5)
    d_shift = float(d[0]) if origin in ("correct_disp", "correct_dispforc") else 0.0
    d, f = _as_curve(pushover, 5, origin=origin)
    n = len(d)

    i_peak = int(np.argmax(f))
    f_peak = float(f[i_peak])
    if f_peak <= 0.0:
        raise ValueError("Peak force must be positive")
    if i_peak >= n - 1:
        raise ValueError("No descending branch: peak is at the last data point")

    k_e = elastic_slope(d[: i_peak + 1], f[: i_peak + 1], f_peak, ref_fraction=elastic_ref_fraction)
    d_max = float(d[-1])
    ok: list[bool] = []

    if not residual_plateau:
        # Drop to zero force: only d_f is needed; d_r and the residual
        # fraction play no part, so they are neither required nor checked.
        if d_f is not None:
            d_f_knot = float(d_f) - d_shift
            if not 0.0 < d_f_knot < d_max:
                raise ValueError(
                    f"Require 0 < d_f < d_max after the origin shift; got "
                    f"d_f={d_f_knot:.6g}, d_max={d_max:.6g}"
                )
        else:
            if not 0.0 < joint_force_fraction < 1.0:
                raise ValueError("Require 0 < joint_force_fraction < 1")
            i_f = _snap_descending_crossing(f, i_peak, joint_force_fraction * f_peak)
            if i_f is None or i_f >= n - 1:
                raise ValueError("No descending crossing of joint_force_fraction * f_peak")
            d_f_knot = float(d[i_f])
        bp = _fit_knots_drop(d, f, k_e, d_f_knot, d_max, f_peak, ok)
        mode = "trilinear-cbf-knots"
    else:
        if d_f is not None and d_r is not None:
            # Knots given directly: no snapping, so they need not be data points.
            d_f_knot = float(d_f) - d_shift
            d_r_knot = float(d_r) - d_shift
            if not 0.0 < d_f_knot < d_r_knot <= d_max:
                raise ValueError(
                    f"Require 0 < d_f < d_r <= d_max after the origin shift; got "
                    f"d_f={d_f_knot:.6g}, d_r={d_r_knot:.6g}, d_max={d_max:.6g}"
                )
        else:
            # The fractions are only meaningful on the snapping path.
            if not 0.0 < residual_force_fraction < joint_force_fraction < 1.0:
                raise ValueError("Require 0 < residual_force_fraction < joint_force_fraction < 1")
            i_f = _snap_descending_crossing(f, i_peak, joint_force_fraction * f_peak)
            if i_f is None:
                raise ValueError("No descending crossing of joint_force_fraction * f_peak")
            i_r = _snap_descending_crossing(f, i_f, residual_force_fraction * f_peak)
            if i_r is None:
                # No residual crossing: d_r falls back to the last data point,
                # the plateau has zero length and the fit has only three
                # branches. Say so, since the result is then not quadrilinear.
                warnings.warn(
                    f"fit_cbf_knots_backbone: the force never falls to "
                    f"residual_force_fraction * f_peak = {residual_force_fraction:g} * f_peak "
                    f"after d_f, so there is no residual plateau and d_r is the last data "
                    f"point. Raise residual_force_fraction (the curve ends at "
                    f"{f[-1] / f_peak:.3g} * f_peak) or pass d_f and d_r explicitly",
                    stacklevel=2,
                )
                i_r = n - 1
            if i_f >= i_r:
                raise ValueError("d_f must precede d_r; choose different fractions")
            d_f_knot = float(d[i_f])
            d_r_knot = float(d[i_r])
        bp = _fit_knots_plateau(d, f, k_e, d_f_knot, d_r_knot, d_max, f_peak, ok)
        mode = "quadrilinear-cbf-knots"

    # Both fits reach d_max, so this range lies inside the data.
    obj = _interval_area_error(d, f, bp, float(bp[1, 0]), d_max)
    return _finish(bp, mode, obj, all(ok), "fit_cbf_knots_backbone")


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


# Interior grid points for the outer search over the yield force, and for the
# drop-mode search over where the drop reaches zero force. Each trial of the
# outer search solves an inner problem exactly, so a coarse grid is enough.
_KNOTS_YIELD_GRID = 20
_KNOTS_DROP_GRID = 40

# The drop-mode search covers zero-force points from 1e-6 to 100 times the
# span [d_f, d_max] beyond d_f, on a log scale: from a near-vertical drop to
# one that loses only 1 % of f_f per span.
_DROP_SPAN_MIN = 1e-6
_DROP_SPAN_MAX = 100.0


def _trapezoid_weights(g: FloatArr) -> FloatArr:
    """Weights ``w`` with ``np.trapezoid(y, g) == w @ y`` for any ``y``."""
    w = np.zeros_like(g)
    h = np.diff(g)
    w[:-1] += h / 2.0
    w[1:] += h / 2.0
    return w


def _knot_grid(d: FloatArr, d_lo: float, d_hi: float, knots: list[float]) -> FloatArr:
    """Integration grid over ``[d_lo, d_hi]``: the range ends, the data and the knots inside.

    The same grid :func:`fitpo._common._interval_area_error` uses, so the area
    errors computed here equal the reported ``objective``.
    """
    kn = np.asarray(knots, dtype=np.float64)
    return np.unique(np.concatenate((
        [d_lo, d_hi], d[(d > d_lo) & (d < d_hi)], kn[(kn > d_lo) & (kn < d_hi)]
    )))


def _fit_knots_plateau(
    d: FloatArr,
    f: FloatArr,
    k_e: float,
    d_f: float,
    d_r: float,
    d_max: float,
    f_peak: float,
    ok: list[bool],
) -> FloatArr:
    r"""Fit ``f_y``, ``f_f`` and ``f_r`` for fixed knots; return the breakpoints.

    For a fixed yield force ``f_y`` (which also fixes ``d_y = f_y / k_e``),
    the fitted curve on ``[d_y, d_max]`` is linear in ``f_f`` and ``f_r``:

    .. math::

        \hat F(d) = f_y\,a(d) + f_f\,b(d) + f_r\,c(d)

    with ``a``, ``b`` and ``c`` the piecewise-linear "hat" functions of the
    knots. The trapezoidal area error is then a weighted sum of absolute
    residuals, i.e. a weighted L1 fit, which a linear program solves exactly.
    Only ``f_y`` is left to a 1-D search. This avoids gradient-based search,
    which stalls on the kinks of the absolute value.

    Bounds: ``0 < d_y < d_f``, ``0 <= f_r <= f_f <= 1.5 f_peak``.
    """
    upper = 1.5 * f_peak
    lp_ok: list[bool] = []

    def solve(f_y: float) -> tuple[float, float, float]:
        d_y = f_y / k_e
        g = _knot_grid(d, d_y, d_max, [d_f, d_r])
        a = np.interp(g, [d_y, d_f], [1.0, 0.0])
        b = np.interp(g, [d_y, d_f, d_r], [0.0, 1.0, 0.0])
        c = np.interp(g, [d_f, d_r], [0.0, 1.0])
        # Forces in units of f_peak and weights in units of the span keep the
        # LP well scaled whatever the units of the data.
        y = (np.interp(g, d, f) - f_y * a) / f_peak
        w = _trapezoid_weights(g) / (d_max - d_y)
        x_f, x_r = _weighted_l1_two(y, b, c, w, upper / f_peak, lp_ok)
        if not np.isfinite(x_f):
            return np.inf, 0.0, 0.0
        f_f, f_r = x_f * f_peak, x_r * f_peak
        bp = np.asarray([[d_y, f_y], [d_f, f_f], [d_r, f_r], [d_max, f_r]])
        return _interval_area_error(d, f, bp, d_y, d_max), f_f, f_r

    f_y, _ = _grid_then_brent(
        lambda x: solve(x)[0], 0.0, min(upper, k_e * d_f), _KNOTS_YIELD_GRID, ok
    )
    _, f_f, f_r = solve(f_y)
    ok.append(all(lp_ok))
    return np.asarray(
        [[0.0, 0.0], [f_y / k_e, f_y], [d_f, f_f], [d_r, f_r], [d_max, f_r]], dtype=np.float64
    )


def _weighted_l1_two(
    y: FloatArr, b: FloatArr, c: FloatArr, w: FloatArr, upper: float, ok: list[bool]
) -> tuple[float, float]:
    r"""Solve the two-variable weighted L1 fit behind the plateau mode.

    Finds ``x_f``, ``x_r`` that minimise :math:`\sum_i w_i |y_i - b_i x_f -
    c_i x_r|` subject to ``0 <= x_r <= x_f <= upper``. Returns ``(nan, nan)``
    if the solver fails.

    The primal problem has a variable per grid point; its dual has the same
    number of variables but only two constraint rows, which the dual simplex
    solver handles several times faster. The dual is

    .. math::

        \max_{\lambda, p, q}\; y^\top \lambda - u (p_f + p_r) \quad
        \text{s.t.}\quad |\lambda_i| \le w_i,\quad
        B^\top \lambda - p - G^\top q \le 0,\quad p, q \ge 0

    with :math:`B = [b, c]`, :math:`u` = ``upper`` and :math:`G = [-1, 1]`
    (the row of ``x_r - x_f <= 0``). The primal solution is the multiplier
    of the two rows of :math:`B^\top \lambda - \dots \le 0`.
    """
    g_row = np.asarray([-1.0, 1.0])
    a_ub = np.hstack([np.vstack([b, c]), -np.eye(2), -g_row[:, None]])
    cost = np.r_[-y, upper, upper, 0.0]
    lower = np.r_[-w, 0.0, 0.0, 0.0]
    high = np.r_[w, np.inf, np.inf, np.inf]
    res = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.zeros(2),
        bounds=np.column_stack([lower, high]),
        method="highs-ds",
    )
    ok.append(bool(res.success))
    if not res.success:
        return np.nan, np.nan
    x_f, x_r = (-float(m) for m in res.ineqlin.marginals)
    # Clip round-off so the shape always satisfies the bounds exactly.
    x_f = min(max(x_f, 0.0), upper)
    x_r = min(max(x_r, 0.0), x_f)
    return x_f, x_r


def _weighted_l1_scale(y: FloatArr, b: FloatArr, w: FloatArr, lo: float, hi: float) -> float:
    """The ``x`` in ``[lo, hi]`` that minimises ``sum(w * |y - x * b|)``.

    Each term is ``w |b| |y / b - x|``, so the minimiser is the weighted
    median of ``y / b`` with weights ``w |b|``. The sum is convex in ``x``,
    so clipping to the bounds keeps it optimal.
    """
    m = (b != 0.0) & (w > 0.0)
    if not bool(m.any()):
        return lo
    r = y[m] / b[m]
    ww = w[m] * np.abs(b[m])
    order = np.argsort(r)
    cum = np.cumsum(ww[order])
    x = float(r[order][int(np.searchsorted(cum, 0.5 * cum[-1]))])
    return min(max(x, lo), hi)


def _fit_knots_drop(
    d: FloatArr,
    f: FloatArr,
    k_e: float,
    d_f: float,
    d_max: float,
    f_peak: float,
    ok: list[bool],
) -> FloatArr:
    """Fit ``f_y``, ``f_f`` and the drop to zero force; return the breakpoints.

    The drop runs from ``(d_f, f_f)`` straight down to zero force at
    ``d_zero``, so its slope is ``-f_f / (d_zero - d_f)``. For fixed ``f_y``
    and ``d_zero`` the fit is linear in ``f_f``, and the best ``f_f`` is a
    weighted median (:func:`_weighted_l1_scale`). ``d_zero`` is found by a
    1-D search on a log scale of ``d_zero - d_f``, and ``f_y`` by an outer
    1-D search. Only the data range ``[d_y, d_max]`` is scored; the fit may
    reach zero beyond ``d_max``.
    """
    upper = 1.5 * f_peak
    span = d_max - d_f

    def best_f_f(f_y: float, d_zero: float) -> tuple[float, float]:
        d_y = f_y / k_e
        g = _knot_grid(d, d_y, d_max, [d_f, d_zero])
        a = np.interp(g, [d_y, d_f], [1.0, 0.0])
        b = np.interp(g, [d_y, d_f, d_zero], [0.0, 1.0, 0.0])
        y = np.interp(g, d, f) - f_y * a
        w = _trapezoid_weights(g)
        f_f = _weighted_l1_scale(y, b, w, 0.0, upper)
        return float(w @ np.abs(y - f_f * b)), f_f

    def drop_end(f_y: float, inner_ok: list[bool]) -> tuple[float, float]:
        z, err = _grid_then_brent(
            lambda z: best_f_f(f_y, d_f + float(np.exp(z)))[0],
            float(np.log(_DROP_SPAN_MIN * span)),
            float(np.log(_DROP_SPAN_MAX * span)),
            _KNOTS_DROP_GRID,
            inner_ok,
        )
        return d_f + float(np.exp(z)), err

    inner_ok: list[bool] = []
    f_y, _ = _grid_then_brent(
        lambda x: drop_end(x, inner_ok)[1], 0.0, min(upper, k_e * d_f), _KNOTS_YIELD_GRID, ok
    )
    d_zero, _ = drop_end(f_y, inner_ok)
    _, f_f = best_f_f(f_y, d_zero)
    ok.append(all(inner_ok))

    rows = [[0.0, 0.0], [f_y / k_e, f_y], [d_f, f_f], [d_zero, 0.0]]
    if d_zero < d_max:
        rows.append([d_max, 0.0])
    return np.asarray(rows, dtype=np.float64)
