"""Fit piecewise-linear positive and negative envelopes to cyclic pushover data."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

FloatArr = npt.NDArray[np.float64]


def fit_envelope(
    cyclic_pushover: npt.ArrayLike,
    *,
    displacement_tolerance: float = 0.01,
    peak_prominence: float | None = None,
) -> tuple[FloatArr, FloatArr]:
    """Fit piecewise-linear positive and negative envelopes to cyclic pushover data.

    Parameters
    ----------
    cyclic_pushover
        Shape (N, 2): col 0 displacement (time-ordered, non-monotone),
        col 1 force.
    displacement_tolerance
        Fraction of max |displacement|. Two cycle peaks within
        ``displacement_tolerance * max(|d|)`` are treated as the same
        amplitude; only the one with the largest |force| is kept.
    peak_prominence
        Absolute prominence passed to ``scipy.signal.find_peaks``. Defaults to
        ``0.01 * max(|d|)``.

    Returns
    -------
    (positive_envelope, negative_envelope)
        Each (M, 2) float64 ndarray of breakpoints starting at ``[0, 0]``. The
        positive envelope has positive displacements/forces; the negative
        envelope has negative displacements/forces.
    """
    arr = np.asarray(cyclic_pushover, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"cyclic_pushover must be shape (N, 2); got {arr.shape}")
    if arr.shape[0] < 3:
        raise ValueError("cyclic_pushover must have at least 3 points")

    d: FloatArr = arr[:, 0]
    f: FloatArr = arr[:, 1]

    d_abs_max = float(np.max(np.abs(d)))
    if d_abs_max <= 0.0:
        raise ValueError("displacement signal has zero range")

    prominence = peak_prominence if peak_prominence is not None else 0.01 * d_abs_max
    abs_tol = displacement_tolerance * d_abs_max

    pos_idx, _ = find_peaks(d, prominence=prominence)
    neg_idx, _ = find_peaks(-d, prominence=prominence)
    pos_idx = pos_idx[d[pos_idx] > 0.0]
    neg_idx = neg_idx[d[neg_idx] < 0.0]

    pos_peaks = np.column_stack([d[pos_idx], f[pos_idx]])
    neg_peaks = np.column_stack([d[neg_idx], f[neg_idx]])

    positive_envelope = _build_envelope(pos_peaks, abs_tol, sign=+1)
    negative_envelope = _build_envelope(neg_peaks, abs_tol, sign=-1)

    return positive_envelope, negative_envelope


def _build_envelope(peaks: FloatArr, abs_tol: float, *, sign: int) -> FloatArr:
    """Group, dedupe, and assemble one side of the envelope.

    For ``sign=+1``: sort peaks by displacement ascending and keep the largest
    force in each near-equal-displacement group.

    For ``sign=-1``: sort peaks by displacement descending (least-negative
    first) and keep the most-negative force in each near-equal-|d| group.

    The returned array is prepended with ``[0, 0]``.
    """
    if peaks.shape[0] == 0:
        return np.asarray([[0.0, 0.0]], dtype=np.float64)

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

    breakpoints = [(0.0, 0.0), *grouped]
    return np.asarray(breakpoints, dtype=np.float64)
