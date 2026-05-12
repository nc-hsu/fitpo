"""Visualise fit_envelope over each CSV in example_cpo_curves/.

Usage:
    python scripts/demo_envelope_fits.py
    python scripts/demo_envelope_fits.py --save scripts/envelope_fits.png
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

from fitpo import fit_envelope  # noqa: E402

CURVES_DIR = REPO_ROOT / "example_cpo_curves"

_COL_HEADERS = "curve            n_pos   n_neg   d_pos_max   f_pos_max   d_neg_min   f_neg_min"


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
        ax.plot(data[:, 0], data[:, 1], color="steelblue", lw=0.4, alpha=0.5, label="cyclic")
        try:
            pos, neg = fit_envelope(data)
            ax.plot(pos[:, 0], pos[:, 1], color="red", marker="o", markersize=4, lw=1.5, label="+ envelope")
            ax.plot(neg[:, 0], neg[:, 1], color="green", marker="o", markersize=4, lw=1.5, label="- envelope")
            d_pos_max = pos[-1, 0] if pos.shape[0] > 1 else 0.0
            f_pos_max = float(np.max(pos[:, 1])) if pos.shape[0] > 1 else 0.0
            d_neg_min = neg[-1, 0] if neg.shape[0] > 1 else 0.0
            f_neg_min = float(np.min(neg[:, 1])) if neg.shape[0] > 1 else 0.0
            print(
                f"{csv_path.stem:<16} "
                f"{pos.shape[0]:<7d} {neg.shape[0]:<7d} "
                f"{d_pos_max:<11.3f} {f_pos_max:<11.3g} "
                f"{d_neg_min:<11.3f} {f_neg_min:<11.3g}"
            )
        except ValueError as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=8)
            print(f"{csv_path.stem:<16} FAILED: {e}")

        ax.set_title(csv_path.stem, fontsize=9)
        ax.set_xlabel("displacement")
        ax.set_ylabel("force")
        ax.axhline(0, color="black", lw=0.3)
        ax.axvline(0, color="black", lw=0.3)
        ax.grid(alpha=0.3)

    for ax in list(axes.flat)[len(csvs):]:
        ax.set_visible(False)

    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("fit_envelope", fontsize=11)
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=120)
        print(f"saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
