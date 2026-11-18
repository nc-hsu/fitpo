"""Synthetic pushover-curve fixtures for tests and viz script."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest


@dataclass(frozen=True)
class Fixture:
    label: str
    curve: np.ndarray
    expected_mode: str | None
    expected_breaks: np.ndarray | None
    raises_for: tuple[str, ...] = field(default_factory=tuple)


def _sample_piecewise(breaks: np.ndarray, n: int = 401) -> tuple[np.ndarray, np.ndarray]:
    d = np.linspace(float(breaks[0, 0]), float(breaks[-1, 0]), n)
    f = np.interp(d, breaks[:, 0], breaks[:, 1])
    return d, f


def _add_noise(f: np.ndarray, f_peak: float, seed: int, fraction: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = f + rng.normal(0.0, fraction * f_peak, size=f.shape)
    # The unloaded first point is exact in real data, so keep it at zero force.
    noisy[0] = f[0]
    return noisy


def build_smooth_quadrilinear() -> Fixture:
    breaks = np.asarray(
        [[0.0, 0.0], [1.0, 100.0], [3.0, 150.0], [6.0, 105.0], [10.0, 105.0]]
    )
    d, f_clean = _sample_piecewise(breaks)
    f = _add_noise(f_clean, f_peak=150.0, seed=42)
    return Fixture(
        label="smooth quadrilinear",
        curve=np.column_stack([d, f]),
        expected_mode=None,
        expected_breaks=breaks,
    )


def build_smooth_trilinear() -> Fixture:
    breaks = np.asarray([[0.0, 0.0], [1.0, 100.0], [3.0, 150.0], [10.0, 0.0]])
    d, f_clean = _sample_piecewise(breaks)
    f = _add_noise(f_clean, f_peak=150.0, seed=43)
    return Fixture(
        label="smooth trilinear",
        curve=np.column_stack([d, f]),
        expected_mode=None,
        expected_breaks=breaks,
    )


def build_bilinear_source() -> Fixture:
    breaks = np.asarray(
        [[0.0, 0.0], [1.0, 100.0], [3.0, 150.0], [6.0, 105.0], [10.0, 105.0]]
    )
    d, f_clean = _sample_piecewise(breaks)
    f = _add_noise(f_clean, f_peak=150.0, seed=44)
    return Fixture(
        label="bilinear fit of quadrilinear",
        curve=np.column_stack([d, f]),
        expected_mode="bilinear",
        expected_breaks=None,
    )


def build_staircase() -> Fixture:
    """Quadrilinear-ish underlying + stair-step drops on descending branch."""
    levels = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 100.0],
            [3.0, 150.0],
            [3.1, 135.0],
            [4.0, 135.0],
            [4.1, 120.0],
            [5.0, 120.0],
            [5.1, 110.0],
            [6.0, 105.0],
            [10.0, 105.0],
        ]
    )
    d, f_clean = _sample_piecewise(levels, n=501)
    f = _add_noise(f_clean, f_peak=150.0, seed=45)
    approx = np.asarray(
        [[0.0, 0.0], [1.0, 100.0], [3.0, 150.0], [6.0, 110.0], [10.0, 110.0]]
    )
    return Fixture(
        label="staircase descending",
        curve=np.column_stack([d, f]),
        expected_mode=None,
        expected_breaks=approx,
    )


def build_peak_at_end() -> Fixture:
    d = np.linspace(0.0, 5.0, 101)
    f_clean = 50.0 * d
    f = _add_noise(f_clean, f_peak=250.0, seed=46)
    return Fixture(
        label="peak at end",
        curve=np.column_stack([d, f]),
        expected_mode=None,
        expected_breaks=None,
        raises_for=("trilinear", "quadrilinear"),
    )


_FRACTURE_KNOTS_D = np.asarray([0.0, 1.0, 2.0, 3.0, 3.2, 4.0, 5.0])
_FRACTURE_KNOTS_F = np.asarray([0.0, 100.0, 180.0, 240.0, 250.0, 80.0, 75.0])


def _fracture_backbone(d_abs: np.ndarray) -> np.ndarray:
    return np.interp(d_abs, _FRACTURE_KNOTS_D, _FRACTURE_KNOTS_F)


def build_fracture_cyclic() -> Fixture:
    """Growing-amplitude cyclic trace with a fracture on the push to +4.

    The positive loading backbone peaks at 250 (d=3.2) then drops sharply to 80
    by d=4 (fracture). The final point is the global max displacement (+5), so
    ``find_peaks`` never flags it. Cycle tips: +1,+2,+3,+4 (interior) and
    -1,-2,-3; the true 250 peak sits between tips, so peak-to-peak misses it.
    """
    targets = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, 5.0]
    d_parts = [np.asarray([0.0])]
    f_parts = [np.asarray([0.0])]
    cur = 0.0
    for k, target in enumerate(targets):
        sign = np.sign(target)
        seg = np.linspace(cur, target, 60)[1:]
        d_parts.append(seg)
        f_parts.append(sign * _fracture_backbone(np.abs(seg)))
        cur = target
        if k != len(targets) - 1:  # unload back to 0, except after the final push
            peak_f = sign * _fracture_backbone(abs(target))
            seg2 = np.linspace(target, 0.0, 60)[1:]
            d_parts.append(seg2)
            f_parts.append(peak_f * (seg2 / target))
            cur = 0.0
    d = np.concatenate(d_parts)
    f = np.concatenate(f_parts)
    return Fixture(
        label="fracture cyclic",
        curve=np.column_stack([d, f]),
        expected_mode=None,
        expected_breaks=None,
    )


def build_postpeak_drop_cyclic() -> Fixture:
    """Cyclic trace whose loading-front plunges below the peak-to-peak chord.

    A high cycle reaches +2 at force 200 *before* fracture, then a degraded low
    cycle pushes out to +4 at only force 55. The two detected tips are (2, 200)
    and (4, 55); between them the loading-front (the +4 push, post-fracture) sits
    far below the connecting chord. So peak-to-peak (and the outward-hull
    adjusted mode) overestimate this descending branch, while simplified-loading-front
    follows the drop. Mirrored on the negative side.
    """
    pushes = [(2.0, 200.0), (-2.0, -200.0), (4.0, 55.0), (-4.0, -55.0)]
    d_parts = [np.asarray([0.0])]
    f_parts = [np.asarray([0.0])]
    for target, peak_f in pushes:
        out = np.linspace(0.0, target, 60)[1:]
        d_parts.append(out)
        f_parts.append(peak_f * (out / target))
        back = np.linspace(target, 0.0, 60)[1:]
        d_parts.append(back)
        f_parts.append(peak_f * (back / target))
    d = np.concatenate(d_parts)
    f = np.concatenate(f_parts)
    return Fixture(
        label="post-peak drop cyclic",
        curve=np.column_stack([d, f]),
        expected_mode=None,
        expected_breaks=None,
    )


ALL_FIXTURE_BUILDERS = [
    build_smooth_quadrilinear,
    build_smooth_trilinear,
    build_bilinear_source,
    build_staircase,
    build_peak_at_end,
]


@pytest.fixture
def smooth_quadrilinear() -> Fixture:
    return build_smooth_quadrilinear()


@pytest.fixture
def smooth_trilinear() -> Fixture:
    return build_smooth_trilinear()


@pytest.fixture
def bilinear_source() -> Fixture:
    return build_bilinear_source()


@pytest.fixture
def staircase() -> Fixture:
    return build_staircase()


@pytest.fixture
def peak_at_end() -> Fixture:
    return build_peak_at_end()


@pytest.fixture
def fracture_cyclic() -> Fixture:
    return build_fracture_cyclic()


@pytest.fixture
def postpeak_drop_cyclic() -> Fixture:
    return build_postpeak_drop_cyclic()
