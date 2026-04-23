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
    return f + rng.normal(0.0, fraction * f_peak, size=f.shape)


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
