"""Visualise fit_piecewise_backbone(mode='bilinear_incl_neg_stiffness') over
each CSV in example_bilinear_po_curves/.

Usage:
    python scripts/demo_bilinear_incl_neg_stiffness_fits.py
    python scripts/demo_bilinear_incl_neg_stiffness_fits.py --save scripts/bilinear_incl_neg_stiffness_fits.png
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

from fitpo import fit_piecewise_backbone  # noqa: E402

CURVES_DIR = REPO_ROOT / "example_bilinear_po_curves"

_COL_HEADERS = "curve                              d_y      f_y      d_u      f_u       k_e        k_2"


def _slope(p0: np.ndarray, p1: np.ndarray) -> float:
    return (p1[1] - p0[1]) / max(p1[0] - p0[0], 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, default=None)
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
        ax.plot(data[:, 0], data[:, 1], color="blue", lw=0.8, label="pushover")
        ax.axhline(0.0, color="grey", lw=0.5)
        try:
            fit = fit_piecewise_backbone(data, mode="bilinear_incl_neg_stiffness")
            ax.plot(fit[:, 0], fit[:, 1], color="red", marker="o", markersize=5, lw=1.5, label="fit")
            k_e = _slope(fit[0], fit[1])
            k_2 = _slope(fit[1], fit[2])
            print(
                f"{csv_path.stem:<32} "
                f"{fit[1, 0]:8.3f} {fit[1, 1]:9.1f} "
                f"{fit[2, 0]:8.3f} {fit[2, 1]:9.1f} "
                f"{k_e:10.1f} {k_2:10.1f}"
            )
        except ValueError as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=8)
            print(f"{csv_path.stem:<32} FAILED: {e}")

        ax.set_title(csv_path.stem, fontsize=9)
        ax.set_xlabel("displacement")
        ax.set_ylabel("force")
        ax.grid(alpha=0.3)

    for ax in list(axes.flat)[len(csvs):]:
        ax.set_visible(False)

    axes.flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("fit_piecewise_backbone(mode='bilinear_incl_neg_stiffness')", fontsize=11)
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=120)
        print(f"saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
