"""Grid of matplotlib plots visualising fit_piecewise_backbone on each test fixture.

Usage:
    python scripts/visualize_tests.py            # plt.show()
    python scripts/visualize_tests.py --save out.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

import conftest as cf  # noqa: E402
from fitpo import fit_piecewise_backbone  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, default=None, help="save PNG path instead of showing")
    args = parser.parse_args()

    fixtures = [b() for b in cf.ALL_FIXTURE_BUILDERS]
    ncols = 3
    nrows = math.ceil(len(fixtures) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax, fx in zip(axes.flat, fixtures, strict=False):
        d = fx.curve[:, 0]
        f = fx.curve[:, 1]
        ax.plot(d, f, color="blue", lw=1.0, label="pushover")
        try:
            fit = fit_piecewise_backbone(fx.curve, mode=fx.expected_mode)
            ax.plot(fit[:, 0], fit[:, 1], color="red", marker="o", lw=1.5, label="fit")
        except ValueError as e:
            ax.text(
                0.5,
                0.5,
                f"raises: {e}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="red",
            )
        mode_str = fx.expected_mode or "auto"
        ax.set_title(f"{fx.label} [{mode_str}]")
        ax.set_xlabel("displacement")
        ax.set_ylabel("force")
        ax.grid(alpha=0.3)

    # hide any unused axes
    for ax in list(axes.flat)[len(fixtures):]:
        ax.set_visible(False)

    axes.flat[0].legend(loc="lower right")
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=120)
        print(f"saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
