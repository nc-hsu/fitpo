---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# The `origin` option

Every backbone fit assumes the curve starts at the unloaded point $(0, 0)$.
The curves in `examples/data/cbf_cusum` start slightly away from it. The
`origin` option (see {doc}`../conventions`) sets what happens.

```{code-cell} python
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
curve = np.loadtxt(DATA / "cbf_cusum" / "po_curve_31.csv", delimiter=",")
curve[0]
```

With the default, `origin="strict"`, the fit refuses the curve and says how
far its first point is from $(0, 0)$:

```{code-cell} python
try:
    fitpo.fit_deluca_backbone(curve)
except ValueError as err:
    print(err)
```

To show the other settings clearly, this example adds an initial force of
100 kN as well, as if the first recorded point were already loaded:

```{code-cell} python
offset = curve.copy()
offset[:, 1] += 100e3
offset[0]
```

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True, facecolor="w")
for ax, origin in zip(axes, ["correct_disp", "correct_dispforc", "as-is"]):
    res = fitpo.fit_deluca_backbone(offset, mode="trilinear-deluca", origin=origin)
    # Plot the curve in the same coordinates as the fit.
    shown = offset.copy()
    if origin != "as-is":
        shown[:, 0] -= offset[0, 0]
    if origin == "correct_dispforc":
        shown[:, 1] -= offset[0, 1]
    ax.plot(shown[:, 0], shown[:, 1] / 1e3, color="0.6", lw=3, label="curve")
    ax.plot(res.fitted_po[:, 0], res.fitted_po[:, 1] / 1e3, "o-", label="trilinear fit")
    ax.set_xlim(-5, 120)
    ax.set_title(f'origin="{origin}"')
    ax.set_xlabel("roof displacement [mm]")
axes[0].set_ylabel("base shear [kN]")
axes[0].legend()
fig.tight_layout()
```

- `"correct_disp"` shifts the displacements only, so the 100 kN offset stays
  in the curve and the fit's elastic line, which always starts at $(0, 0)$,
  has to rise through it.
- `"correct_dispforc"` subtracts the whole first point, so the curve starts
  at $(0, 0)$ again.
- `"as-is"` leaves the curve unchanged; the gap between $(0, 0)$ and the first
  point counts as elastic.

Which setting is right depends on why the curve does not start at
$(0, 0)$. If the unloaded point was simply not recorded, adding the row
`[0, 0]` to the data is often the best correction.
