"""Positive and negative envelopes of cyclic pushover data.

A cyclic pushover trace goes back and forth in displacement, so it has no
single backbone. This module extracts one piecewise-linear envelope for the
positive side and one for the negative side, using one of several rules for
which trace points to keep (cycle peaks, the loading front, displacement
bins, or simplifications of these). The envelopes can then be fitted with
the backbone functions like a monotonic pushover curve.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from fitpo._common import _as_curve

FloatArr = npt.NDArray[np.float64]

EnvelopeMode = Literal[
    "peak-to-peak",
    "loading-front",
    "displacement-bins",
    "adjusted-peak-to-peak",
    "simplified-loading-front",
]
_MODES: tuple[str, ...] = (
    "peak-to-peak",
    "loading-front",
    "displacement-bins",
    "adjusted-peak-to-peak",
    "simplified-loading-front",
)


def fit_envelope(
    cyclic_pushover: npt.ArrayLike,
    *,
    mode: EnvelopeMode = "peak-to-peak",
    bin_size: float = 1.0,
    simplify_tolerance: float = 0.05,
    include_final_peak: bool = False,
    close_at_zero: bool = False,
    displacement_tolerance: float = 0.01,
    peak_prominence: float | None = None,
) -> tuple[FloatArr, FloatArr]:
    """Fit piecewise-linear positive and negative envelopes to cyclic pushover data.

    Parameters
    ----------
    cyclic_pushover
        Shape (N, 2): col 0 displacement (time-ordered, non-monotone),
        col 1 force.
    mode
        How the outer backbone is traced:

        - ``"peak-to-peak"`` (default): detect cycle peaks and draw straight
          lines between them. Fast, but bridges over any mid-loading force
          drop (fracture).
        - ``"loading-front"``: keep every trace point whose |displacement|
          exceeds all earlier points in time (the loading frontier). Follows
          the real loading branch, so fracture drops are captured.
        - ``"displacement-bins"``: bin the displacement range into steps of
          ``bin_size`` and keep the real ``(d, f)`` point of extreme force in
          each bin. The outermost upper boundary of the trace; captures
          fracture drops and later-cycle re-crossings at higher force.
        - ``"adjusted-peak-to-peak"``: the ``"peak-to-peak"`` skeleton with the
          minimum number of ``"loading-front"`` points inserted to cover any
          section that protrudes outside the peak-to-peak chords. Captures a
          fracture peak/drop sitting between cycle tips without the saw teeth of
          the two outer-trace modes. Only pulls the envelope outward, so it can
          still overestimate a sheer drop *below* a chord.
        - ``"simplified-loading-front"``: the ``"peak-to-peak"`` skeleton with each
          chord segment replaced by a Ramer-Douglas-Peucker simplification of
          the ``"loading-front"`` within it. Keeps significant deviations both
          above *and* below the chord (``simplify_tolerance``), so it captures a
          fracture apex like ``"adjusted-peak-to-peak"`` but additionally follows
          the sheer post-peak drop instead of overestimating it, while still
          dropping small saw-tooth noise.
    bin_size
        Displacement-bin width (same units as displacement, default ``1.0``).
        Only used when ``mode="displacement-bins"``.
    simplify_tolerance
        Deviation threshold for ``mode="simplified-loading-front"``, as a fraction of
        the side's max |force| (default ``0.05``). Loading-front points whose
        force deviates from the local chord by more than
        ``simplify_tolerance * max|force|`` are kept; smaller wiggles are dropped.
    include_final_peak
        If ``True`` and the last trace point is the largest |displacement| on
        its side, treat that final point as a peak/breakpoint. Mainly relevant
        for ``"peak-to-peak"`` (``find_peaks`` never flags an endpoint); the
        other modes include it naturally.
    close_at_zero
        If ``True``, append a final breakpoint at ``(extreme_displacement, 0)``
        so the envelope drops to zero force at the largest displacement.
    displacement_tolerance
        Fraction of max |displacement|. Two cycle peaks within
        ``displacement_tolerance * max(|d|)`` are treated as the same
        amplitude; only the one with the largest |force| is kept. Also the
        tolerance for detecting whether the final point is the side extreme.
    peak_prominence
        Absolute prominence passed to ``scipy.signal.find_peaks`` (only used by
        ``"peak-to-peak"``). Defaults to ``0.01 * max(|d|)``.

    Returns
    -------
    (positive_envelope, negative_envelope)
        Each (M, 2) float64 ndarray of breakpoints starting at ``[0, 0]``. The
        positive envelope has positive displacements/forces; the negative
        envelope has negative displacements/forces.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}; got {mode!r}")
    if mode == "displacement-bins" and bin_size <= 0.0:
        raise ValueError(f"bin_size must be positive; got {bin_size}")

    # Cyclic data goes back and forth in displacement by design, so only the
    # shape and NaN/inf are checked here, not monotonicity.
    d, f = _as_curve(cyclic_pushover, 3, name="cyclic_pushover", monotone=False)

    d_abs_max = float(np.max(np.abs(d)))
    if d_abs_max <= 0.0:
        raise ValueError("displacement signal has zero range")

    prominence = peak_prominence if peak_prominence is not None else 0.01 * d_abs_max
    abs_tol = displacement_tolerance * d_abs_max

    positive_envelope = _build_side(
        d, f,
        sign=+1,
        mode=mode,
        bin_size=bin_size,
        simplify_tolerance=simplify_tolerance,
        prominence=prominence,
        abs_tol=abs_tol,
        d_abs_max=d_abs_max,
        include_final_peak=include_final_peak,
        close_at_zero=close_at_zero,
    )
    negative_envelope = _build_side(
        d, f,
        sign=-1,
        mode=mode,
        bin_size=bin_size,
        simplify_tolerance=simplify_tolerance,
        prominence=prominence,
        abs_tol=abs_tol,
        d_abs_max=d_abs_max,
        include_final_peak=include_final_peak,
        close_at_zero=close_at_zero,
    )

    return positive_envelope, negative_envelope


def _build_side(
    d: FloatArr,
    f: FloatArr,
    *,
    sign: int,
    mode: str,
    bin_size: float,
    simplify_tolerance: float,
    prominence: float,
    abs_tol: float,
    d_abs_max: float,
    include_final_peak: bool,
    close_at_zero: bool,
) -> FloatArr:
    """Assemble one side (positive or negative) of the envelope.

    Selects the per-mode breakpoint list (excluding the leading origin), then
    prepends ``[0, 0]`` and applies the ``include_final_peak`` and
    ``close_at_zero`` options.
    """
    if mode == "loading-front":
        points = _loading_front_points(d, f, sign=sign)
    elif mode == "displacement-bins":
        points = _binned_points(d, f, sign=sign, bin_size=bin_size)
    elif mode == "adjusted-peak-to-peak":
        points = _adjusted_peak_points(
            d, f, sign=sign, prominence=prominence, abs_tol=abs_tol
        )
    elif mode == "simplified-loading-front":
        points = _simplified_front_points(
            d, f, sign=sign, prominence=prominence, abs_tol=abs_tol,
            simplify_tolerance=simplify_tolerance,
        )
    else:  # peak-to-peak
        points = _peak_points(d, f, sign=sign, prominence=prominence, abs_tol=abs_tol)

    breakpoints: list[tuple[float, float]] = [(0.0, 0.0), *points]

    if include_final_peak:
        _apply_final_peak(breakpoints, d, f, sign=sign, d_abs_max=d_abs_max, abs_tol=abs_tol)

    if close_at_zero and len(breakpoints) > 1:
        d_extreme = breakpoints[-1][0]
        breakpoints.append((d_extreme, 0.0))

    return np.asarray(breakpoints, dtype=np.float64)


def _peak_points(
    d: FloatArr, f: FloatArr, *, sign: int, prominence: float, abs_tol: float
) -> list[tuple[float, float]]:
    """Peak-to-peak breakpoints: detected cycle peaks, grouped and deduped.

    For ``sign=+1``: displacement local maxima sorted ascending, keeping the
    largest force in each near-equal-displacement group. For ``sign=-1``:
    displacement local minima sorted descending (least-negative first), keeping
    the most-negative force in each near-equal-|d| group.
    """
    idx, _ = find_peaks(sign * d, prominence=prominence)
    idx = idx[sign * d[idx] > 0.0]
    if idx.shape[0] == 0:
        return []

    peaks = np.column_stack([d[idx], f[idx]])
    order = np.argsort(peaks[:, 0]) if sign > 0 else np.argsort(-peaks[:, 0])
    ordered = peaks[order]

    grouped: list[tuple[float, float]] = []
    cur_d = float(ordered[0, 0])
    cur_f = float(ordered[0, 1])
    for i in range(1, ordered.shape[0]):
        di = float(ordered[i, 0])
        fi = float(ordered[i, 1])
        if abs(di - cur_d) < abs_tol:
            better = fi > cur_f if sign > 0 else fi < cur_f
            if better:
                cur_d, cur_f = di, fi
        else:
            grouped.append((cur_d, cur_f))
            cur_d, cur_f = di, fi
    grouped.append((cur_d, cur_f))
    return grouped


def _loading_front_points(
    d: FloatArr, f: FloatArr, *, sign: int
) -> list[tuple[float, float]]:
    """Record-high frontier: points whose |displacement| beats all earlier ones.

    Walks the trace in time order, keeping each point that pushes past the
    running extreme displacement on its side. Points come out sorted outward.
    """
    points: list[tuple[float, float]] = []
    running = 0.0
    for i in range(d.shape[0]):
        di = float(d[i])
        if sign * di > running:
            running = sign * di
            points.append((di, float(f[i])))
    return points


def _binned_points(
    d: FloatArr, f: FloatArr, *, sign: int, bin_size: float
) -> list[tuple[float, float]]:
    """Per-bin extreme-force points: the outermost upper boundary of the trace.

    Same-sign points are grouped into displacement bins of width ``bin_size``;
    each bin contributes the real ``(d, f)`` point of extreme force (max for
    ``sign=+1``, min for ``sign=-1``). Ordered by |displacement| ascending.
    """
    mask = sign * d > 0.0
    if not np.any(mask):
        return []

    dm = d[mask]
    fm = f[mask]
    bins = np.floor(np.abs(dm) / bin_size).astype(np.int64)

    best: dict[int, tuple[float, float]] = {}
    for b, di, fi in zip(bins.tolist(), dm.tolist(), fm.tolist(), strict=True):
        cur = best.get(b)
        if cur is None or (fi > cur[1] if sign > 0 else fi < cur[1]):
            best[b] = (di, fi)

    return [best[b] for b in sorted(best)]


def _adjusted_peak_points(
    d: FloatArr, f: FloatArr, *, sign: int, prominence: float, abs_tol: float
) -> list[tuple[float, float]]:
    """Peak-to-peak skeleton with protruding loading-front points inserted.

    Walks each peak-to-peak segment ``[P_a, P_b]`` (including the origin as the
    innermost anchor) and inserts the interior vertices of the outward convex
    hull of the loading-front points falling strictly inside that segment. This
    is the fewest real points needed so straight segments stay outboard of every
    protruding point, so a fracture peak between cycle tips is captured while
    saw teeth below the chords are dropped.
    """
    peak_pts = _peak_points(d, f, sign=sign, prominence=prominence, abs_tol=abs_tol)
    skeleton = [(0.0, 0.0), *peak_pts]
    if len(skeleton) < 2:
        return peak_pts

    front = _loading_front_points(d, f, sign=sign)

    result: list[tuple[float, float]] = []
    for (d_a, f_a), (d_b, f_b) in zip(skeleton[:-1], skeleton[1:], strict=False):
        lo, hi = (d_a, d_b) if d_a <= d_b else (d_b, d_a)
        inside = [(pd, pf) for (pd, pf) in front if lo < pd < hi]
        seg = [(d_a, f_a), *inside, (d_b, f_b)]
        hull = _outward_hull(seg, sign=sign)
        result.extend(hull[1:-1])  # interior vertices only; endpoints added below
        if (d_b, f_b) != (0.0, 0.0):
            result.append((d_b, f_b))
    return result


def _outward_hull(
    points: list[tuple[float, float]], *, sign: int
) -> list[tuple[float, float]]:
    """Outward convex hull of ``points`` sorted by |displacement| ascending.

    Upper hull for ``sign>0`` (keeps points of maximal force), lower hull for
    ``sign<0`` (minimal force). Monotone-chain; endpoints are always retained.
    """
    ordered = sorted(points, key=lambda p: sign * p[0])
    if len(ordered) <= 2:
        return ordered

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # The cross product is invariant under the (d, f) -> (-d, -f) mirror that maps
    # the negative side onto the positive side, so the pop condition is identical
    # for both signs once ``ordered`` is sorted outward; only the sort key uses sign.
    hull: list[tuple[float, float]] = []
    for p in ordered:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) >= 0.0:
            hull.pop()
        hull.append(p)
    return hull


def _simplified_front_points(
    d: FloatArr,
    f: FloatArr,
    *,
    sign: int,
    prominence: float,
    abs_tol: float,
    simplify_tolerance: float,
) -> list[tuple[float, float]]:
    """Peak-to-peak skeleton with each chord simplified by an RDP of the front.

    Like ``_adjusted_peak_points`` but, instead of the outward convex hull, each
    peak-to-peak segment is replaced by a Ramer-Douglas-Peucker simplification of
    the loading-front sub-polyline inside it. RDP keeps significant deviations
    both above *and* below the chord, so a fracture apex is captured (as with the
    hull) and the sheer post-peak drop is followed rather than overestimated,
    while small saw-tooth wiggles below ``tol`` are dropped.
    """
    peak_pts = _peak_points(d, f, sign=sign, prominence=prominence, abs_tol=abs_tol)
    skeleton = [(0.0, 0.0), *peak_pts]
    if len(skeleton) < 2:
        return peak_pts

    front = _loading_front_points(d, f, sign=sign)
    if not front:
        return peak_pts
    tol = simplify_tolerance * max(abs(pf) for _, pf in front)

    result: list[tuple[float, float]] = []
    for (d_a, f_a), (d_b, f_b) in zip(skeleton[:-1], skeleton[1:], strict=False):
        lo, hi = (d_a, d_b) if d_a <= d_b else (d_b, d_a)
        inside = [(pd, pf) for (pd, pf) in front if lo < pd < hi]
        seg = sorted([(d_a, f_a), *inside, (d_b, f_b)], key=lambda p: sign * p[0])
        kept = _rdp_simplify(seg, tol)
        result.extend(kept[1:-1])  # interior vertices only; endpoints added below
        if (d_b, f_b) != (0.0, 0.0):
            result.append((d_b, f_b))
    return result


def _rdp_simplify(
    points: list[tuple[float, float]], tol: float
) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification, keeping the endpoints.

    ``points`` are ordered outward (by displacement). Deviation is measured
    vertically in force from the chord ``points[0] -> points[-1]`` (the
    sub-polyline is a function of displacement, so vertical distance is
    scale-appropriate and sign-symmetric). Interior points whose |deviation|
    exceeds ``tol`` are kept recursively; the rest are dropped.
    """
    if len(points) <= 2:
        return points

    d0, f0 = points[0]
    dn, fn = points[-1]
    span = dn - d0

    max_dev = -1.0
    split = 0
    for i in range(1, len(points) - 1):
        di, fi = points[i]
        chord_f = f0 + (fn - f0) * (di - d0) / span if span != 0.0 else f0
        dev = abs(fi - chord_f)
        if dev > max_dev:
            max_dev = dev
            split = i

    if max_dev <= tol:
        return [points[0], points[-1]]

    left = _rdp_simplify(points[: split + 1], tol)
    right = _rdp_simplify(points[split:], tol)
    return left[:-1] + right  # drop the duplicated join point


def _apply_final_peak(
    breakpoints: list[tuple[float, float]],
    d: FloatArr,
    f: FloatArr,
    *,
    sign: int,
    d_abs_max: float,
    abs_tol: float,
) -> None:
    """Ensure the final trace point is a breakpoint when it is the side extreme.

    Mutates ``breakpoints`` in place. If the last trace sample is within
    ``abs_tol`` of the side's extreme displacement, append it (unless it is
    already the last breakpoint).
    """
    d_last = float(d[-1])
    if sign * d_last <= 0.0:
        return
    if abs(sign * d_last - d_abs_max) > abs_tol:
        return
    point = (d_last, float(f[-1]))
    if breakpoints[-1] != point:
        breakpoints.append(point)
