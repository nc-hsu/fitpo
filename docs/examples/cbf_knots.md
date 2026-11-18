---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# CBF fit with fixed knots

{func}`fitpo.fit_cbf_knots_backbone` places the two knots of the fracture
drop at force fractions of the peak (or at given displacements) and fits all
knot forces together (see {doc}`../method`). It works on sparse envelopes,
like the twelve-point curves used here from the example data (see
{doc}`../data`).

```{code-cell} python
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fitpo

# The example data live in the repository's examples/data folder; find it from
# wherever the page is built.
DATA = next(
    p / "examples" / "data"
    for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "examples" / "data").is_dir()
)
paths = sorted((DATA / "cbf_knots").glob("*.csv"))
```

## With and without the residual plateau

By default the fit ends on a horizontal residual plateau
(`"quadrilinear-cbf-knots"`). With `residual_plateau=False` the drop instead
continues straight down to zero force (`"trilinear-cbf-knots"`). Warnings are
collected and printed below the figure; every curve here falls below 5 % of
its peak force, so none is expected.

```{code-cell} python
fig, axes = plt.subplots(3, 3, figsize=(13, 10), facecolor="w")
messages = []
for ax, path in zip(axes.flat, paths):
    curve = np.loadtxt(path, delimiter=",")
    ax.plot(curve[:, 0], curve[:, 1] / 1e3, "x-", color="0.6", lw=2)
    for plateau, style in ((True, "o-"), (False, "s--")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = fitpo.fit_cbf_knots_backbone(curve, residual_plateau=plateau)
        messages += [f"{path.stem}: {w.message}" for w in caught]
        ax.plot(res.fitted_po[:, 0], res.fitted_po[:, 1] / 1e3, style, label=res.mode)
    ax.set_title(path.stem, fontsize=10)
axes[0, 0].legend(fontsize=8)
for ax in axes[2]:
    ax.set_xlabel("roof displacement [mm]")
for ax in axes[:, 0]:
    ax.set_ylabel("base shear [kN]")
fig.tight_layout()
```

```{code-cell} python
print(f"{len(messages)} warnings")
for m in messages:
    print(m)
```

## When the curve never reaches the residual level

The force-fraction knots need the curve to cross both levels after the peak.
The densely sampled curve `cbf_cusum/po_curve_71.csv` ends at about a third
of its peak force, so it never falls to the default residual level of 5 %.
The residual knot then becomes the last data point, the plateau has no
length, the fit has only three branches, and it warns. Raising
`residual_force_fraction` above the level where the curve ends gives the
plateau room again. (This curve starts slightly away from $(0, 0)$, hence
`origin="correct_disp"`.)

```{code-cell} python
curve = np.loadtxt(DATA / "cbf_cusum" / "po_curve_71.csv", delimiter=",")
print(f"the curve ends at {curve[-1, 1] / curve[:, 1].max():.2f} of its peak force")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res_default = fitpo.fit_cbf_knots_backbone(curve, origin="correct_disp")
print("warning:", caught[0].message)
res_raised = fitpo.fit_cbf_knots_backbone(
    curve, residual_force_fraction=0.40, origin="correct_disp"
)

fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="w")
ax.plot(curve[:, 0] - curve[0, 0], curve[:, 1] / 1e3, color="0.6", lw=3, label="curve")
for res, label, style in (
    (res_default, "residual_force_fraction=0.05", "o--"),
    (res_raised, "residual_force_fraction=0.40", "s-"),
):
    ax.plot(res.fitted_po[:, 0], res.fitted_po[:, 1] / 1e3, style,
            label=f"{label} ({len(res.fitted_po)} rows)")
ax.set_xlabel("roof displacement [mm]")
ax.set_ylabel("base shear [kN]")
ax.legend()
fig.tight_layout()
```

## Knots at given displacements

The knots can also be given directly as displacements (in the coordinates
of the input curve). They need not be data points, and the force fractions
are then ignored:

```{code-cell} python
curve = np.loadtxt(DATA / "cbf_knots" / "3s_cbf_dc2_61_env_br.csv", delimiter=",")
res = fitpo.fit_cbf_knots_backbone(curve, d_f=90.0, d_r=125.0)
res.fitted_po.round(1)
```
