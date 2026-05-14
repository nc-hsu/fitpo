"""Visualise fit_tetralinear_backbone over each CSV in example_tetralinear_type_2/.

Usage:
    python scripts/demo_tetralinear_fits.py
    python scripts/demo_tetralinear_fits.py --save scripts/tetralinear_fits.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fitpo import fit_tetralinear_backbone  # noqa: E402

CURVES_DIR = REPO_ROOT / "example_tetralinear_type_2"

_COL_HEADERS = "curve            d_y      f_y     d_joint  f_joint    d_r2     f_r2      k_e       k_pb      k_r"


def _k_e(breaks: np.ndarray) -> float:
    d0, f0 = breaks[0]
    d1, f1 = breaks[1]
    return (f1 - f0) / max(d1 - d0, 1e-12)


def _k_pb(breaks: np.ndarray) -> float:
    d1, f1 = breaks[1]
    d2, f2 = breaks[2]
    return (f2 - f1) / max(d2 - d1, 1e-12)


def _k_r(breaks: np.ndarray) -> float:
    d3, f3 = breaks[3]
    d4, f4 = breaks[4]
    return (f4 - f3) / max(d4 - d3, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument(
        "--joint-force-fraction",
        type=float,
        default=0.80,
        help="Target force fraction for d_f snapping (default 0.80).",
    )
    parser.add_argument(
        "--residual-force-fraction",
        type=float,
        default=0.05,
        help="Target force fraction for d_r snapping (default 0.05).",
    )
    parser.add_argument(
        "--collapse-residual",
        action="store_true",
        help="Extend segment 3 to f=0 and drop the horizontal residual.",
    )
    args = parser.parse_args()

    csvs = sorted(CURVES_DIR.glob("*.csv"))
    if not csvs:
        sys.exit(f"No CSV files found in {CURVES_DIR}")

    ncols = 3
    nrows = math.ceil(len(csvs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    print(_COL_HEADERS)

    for ax, csv_path in zip(axes.flat, csvs, strict=False):
        data = np.loadtxt(csv_path, delimiter=",")
        ax.plot(data[:, 0], data[:, 1], color="blue", lw=0.8, marker=".",
                markersize=4, label="pushover")
        try:
            fit = fit_tetralinear_backbone(
                data,
                joint_force_fraction=args.joint_force_fraction,
                residual_force_fraction=args.residual_force_fraction,
                collapse_residual=args.collapse_residual,
            )
            ax.plot(fit[:, 0], fit[:, 1], color="red", marker="o",
                    markersize=5, lw=1.5, label="fit")
            ke = _k_e(fit)
            kpb = _k_pb(fit)
            kr = _k_r(fit)
            print(
                f"{csv_path.stem:<16} "
                f"{fit[1,0]:8.3f} {fit[1,1]:8.1f} "
                f"{fit[2,0]:8.3f} {fit[2,1]:8.1f} "
                f"{fit[3,0]:8.3f} {fit[3,1]:8.1f} "
                f"{ke:9.1f} {kpb:9.1f} {kr:9.1f}"
            )
        except ValueError as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=8)
            print(f"{csv_path.stem:<16} FAILED: {e}")

        ax.set_title(csv_path.stem, fontsize=9)
        ax.set_xlabel("displacement")
        ax.set_ylabel("force")
        ax.grid(alpha=0.3)

    for ax in list(axes.flat)[len(csvs):]:
        ax.set_visible(False)

    axes.flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"fit_tetralinear_backbone (joint={args.joint_force_fraction}, "
        f"residual={args.residual_force_fraction}"
        f"{', collapse_residual' if args.collapse_residual else ''})",
        fontsize=11,
    )
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=120)
        print(f"saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
