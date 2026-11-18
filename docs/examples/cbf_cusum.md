---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# CBF fit with CUSUM fracture detection

{func}`fitpo.fit_cbf_cusum_backbone` finds the brace-fracture drop of a
concentrically braced frame's pushover curve automatically, with a CUSUM
change detector on the slope of the curve (see {doc}`../method`). The nine
curves are from the example data (see {doc}`../data`); they start slightly
away from $(0, 0)$, so `origin="correct_disp"` shifts the displacements.

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
paths = sorted((DATA / "cbf_cusum").glob("*.csv"))
[p.stem for p in paths]
```

```{code-cell} python
fig, axes = plt.subplots(3, 3, figsize=(13, 10), facecolor="w")
for ax, path in zip(axes.flat, paths):
    curve = np.loadtxt(path, delimiter=",")
    res = fitpo.fit_cbf_cusum_backbone(curve, origin="correct_disp")
    ax.plot(curve[:, 0] - curve[0, 0], curve[:, 1] / 1e3, color="0.6", lw=3)
    ax.plot(res.fitted_po[:, 0], res.fitted_po[:, 1] / 1e3, "o-", color="tab:green")
    ax.set_title(path.stem, fontsize=10)
for ax in axes[2]:
    ax.set_xlabel("roof displacement [mm]")
for ax in axes[:, 0]:
    ax.set_ylabel("base shear [kN]")
fig.tight_layout()
```

The five rows of `fitted_po` are the origin, the yield point, the joint at
the start of the fracture drop, the start of the residual branch and the
end of the curve:

```{code-cell} python
res = fitpo.fit_cbf_cusum_backbone(np.loadtxt(paths[0], delimiter=","), origin="correct_disp")
res.fitted_po.round(1)
```

CUSUM needs enough samples around the drop to see the change in slope. On
sparse envelopes, such as those in {doc}`cbf_knots`, it can find no fracture
and raises an error; use {func}`fitpo.fit_cbf_knots_backbone` there.
