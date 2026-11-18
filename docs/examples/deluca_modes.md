---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# De Luca fits: the four modes

{func}`fitpo.fit_deluca_backbone` fits one curve in four ways. This page
applies each mode to a pushover curve of a braced frame from the example data
(see {doc}`../data`). The curve starts slightly away from $(0, 0)$, so the
displacements are shifted with `origin="correct_disp"`.

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
curve = np.loadtxt(DATA / "cbf_cusum" / "po_curve_71.csv", delimiter=",")
d = curve[:, 0] - curve[0, 0]  # the same shift as origin="correct_disp"
f = curve[:, 1]
```

```{code-cell} python
modes = ["bilinear-deluca", "trilinear-deluca", "quadrilinear-deluca", "auto"]
fits = {m: fitpo.fit_deluca_backbone(curve, mode=m, origin="correct_disp") for m in modes}

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, facecolor="w")
for ax, (m, res) in zip(axes.flat, fits.items()):
    ax.plot(d, f / 1e3, color="0.6", lw=3, label="pushover curve")
    ax.plot(res.fitted_po[:, 0], res.fitted_po[:, 1] / 1e3, "o-", color="tab:blue",
            label=res.mode)
    ax.set_title(f'mode="{m}"')
    ax.legend(loc="lower right")
for ax in axes[1]:
    ax.set_xlabel("roof displacement [mm]")
for ax in axes[:, 0]:
    ax.set_ylabel("base shear [kN]")
fig.tight_layout()
```

`"auto"` chose the shape named in its legend. The area errors over the whole
curve show what the extra segments buy:

```{code-cell} python
for m, res in fits.items():
    err = fitpo.area_error(curve, res.fitted_po, normalised=True, origin="correct_disp")
    print(f"{m:20s} -> {res.mode:20s} rows={len(res.fitted_po)}  normalised area error={err:.4f}")
```

`bilinear-deluca` ignores the loss of strength entirely, while the
trilinear and quadrilinear shapes follow it. Note that none of the De Luca
shapes can follow the sudden drop at the end of this curve, where a brace
fractures; {doc}`deluca_vs_cbf` compares them with the fits made for braced
frames.
