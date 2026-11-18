---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# De Luca vs CBF fits on a braced-frame curve

This page fits three braced-frame pushover curves from the example data (see
{doc}`../data`) with the three De Luca modes and with the two fits made for
concentrically braced frames, and compares their area errors over the whole
curve.

```{code-cell} python
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown

import fitpo

# The example data live in the repository's examples/data folder; find it from
# wherever the page is built.
DATA = next(
    p / "examples" / "data"
    for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "examples" / "data").is_dir()
)
NAMES = ["po_curve_10", "po_curve_61", "po_curve_71"]
ORIGIN = "correct_disp"  # the curves start slightly away from (0, 0)


def all_fits(curve):
    """The five fits compared on this page, keyed by a short label."""
    fits = {}
    for mode in ("trilinear-deluca", "quadrilinear-deluca", "auto"):
        fits[f"De Luca {mode}"] = fitpo.fit_deluca_backbone(curve, mode=mode, origin=ORIGIN)
    fits["CBF CUSUM"] = fitpo.fit_cbf_cusum_backbone(curve, origin=ORIGIN)
    # Some of these curves never fall to 5 % of the peak, so the knots fit
    # warns that it has no residual plateau; that is expected here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits["CBF knots"] = fitpo.fit_cbf_knots_backbone(curve, origin=ORIGIN)
    return fits


curves = {n: np.loadtxt(DATA / "cbf_cusum" / f"{n}.csv", delimiter=",") for n in NAMES}
results = {n: all_fits(c) for n, c in curves.items()}
```

```{code-cell} python
styles = {
    "De Luca trilinear-deluca": ("tab:blue", "-"),
    "De Luca quadrilinear-deluca": ("tab:purple", "-"),
    "De Luca auto": ("tab:cyan", ":"),
    "CBF CUSUM": ("tab:green", "--"),
    "CBF knots": ("tab:red", "--"),
}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor="w")
for ax, (name, curve) in zip(axes, curves.items()):
    ax.plot(curve[:, 0] - curve[0, 0], curve[:, 1] / 1e3, color="0.75", lw=4, label="curve")
    for label, res in results[name].items():
        color, ls = styles[label]
        ax.plot(res.fitted_po[:, 0], res.fitted_po[:, 1] / 1e3, ls, color=color,
                marker="o", ms=3, label=label)
    ax.set_title(name)
    ax.set_xlabel("roof displacement [mm]")
axes[0].set_ylabel("base shear [kN]")
axes[0].legend(fontsize=8)
fig.tight_layout()
```

The area errors use {func}`fitpo.area_error`, which integrates over the whole
curve for every fit, so the shapes can be compared fairly. The normalised
error divides by the area under the curve.

```{code-cell} python
rows = ["| curve | fit | rows | area error [kN mm] | normalised |", "|---|---|---|---|---|"]
for name, curve in curves.items():
    for label, res in results[name].items():
        a = fitpo.area_error(curve, res.fitted_po, origin=ORIGIN)
        a_n = fitpo.area_error(curve, res.fitted_po, origin=ORIGIN, normalised=True)
        rows.append(f"| {name} | {label} | {len(res.fitted_po)} | {a / 1e3:,.0f} | {a_n:.3f} |")
Markdown("\n".join(rows))
```

## Why the De Luca rules fit braced frames poorly

The De Luca et al. (2013) rules describe a capacity curve that hardens to a
peak and then softens gradually, possibly to a residual plateau. A braced
frame behaves differently:

- **Buckling comes early and suddenly.** The compression braces buckle close
  to the peak, so the hardening branch is short or missing, and the strength
  starts to fall almost at once.
- **The post-buckling decline is long and gentle.** The frame keeps most of
  its strength while the tension braces yield, so a single softening line
  from the peak has to choose between this gentle decline and the drop that
  follows.
- **Fracture is a sudden drop late in the curve.** When a brace fractures,
  the strength falls sharply over a small displacement and then continues on
  a lower residual branch. The De Luca softening segment is a single straight
  line from the peak (and the plateau is horizontal), so it cannot follow a
  gentle decline followed by a sharp drop: it cuts across the drop and
  misplaces both branches.

The CBF fits have a segment for each of these stages, so they follow the
curve more closely, as the table shows.
