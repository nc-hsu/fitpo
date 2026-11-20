---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Quick start

A pushover curve is an `(N, 2)` array: displacement in column 0 and force in
column 1, starting at the unloaded point `(0, 0)`. Any consistent units work;
the examples use mm and N.

```{code-cell} python
import numpy as np
import fitpo

# A curve that rises, hardens to a peak, softens and settles on a plateau.
d = np.linspace(0.0, 120.0, 241)
f = np.interp(d, [0, 10, 40, 80, 120], [0, 1000, 1200, 600, 600])
curve = np.column_stack([d, f])
```

Fit a backbone with the De Luca et al. (2013) rules. In the default `"auto"`
mode, the fit chooses between a trilinear and a quadrilinear shape:

```{code-cell} python
res = fitpo.fit_deluca_backbone(curve)
res.mode
```

Every backbone function returns a {class}`~fitpo.FitResult`. Its
`fitted_po` holds the breakpoints of the backbone, one `[d, F]` row each:

```{code-cell} python
res.fitted_po.round(1)
```

`objective` is the value of the objective the fit minimised, and `converged` says
whether every optimiser reported success:

```{code-cell} python
res.objective, res.converged
```

To compare fits of different shapes, use {func}`~fitpo.area_error`, which
always integrates over the whole curve. With `normalised=True` it is divided
by the area under the curve:

```{code-cell} python
tri = fitpo.fit_deluca_backbone(curve, mode="trilinear-deluca")
for r in (res, tri):
    print(r.mode, round(fitpo.area_error(curve, r.fitted_po, normalised=True), 4))
```

Plot the curve and the two fits:

```{code-cell} python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4), facecolor="w")
ax.plot(d, f, color="0.6", lw=3, label="pushover curve")
for r in (res, tri):
    ax.plot(r.fitted_po[:, 0], r.fitted_po[:, 1], "o-", label=r.mode)
ax.set_xlabel("roof displacement [mm]")
ax.set_ylabel("base shear [N]")
ax.legend();
```

Next steps:

- {doc}`method` explains every fit and its rules;
- {doc}`conventions` describes inputs, outputs and the `origin` option;
- the examples apply each function to real pushover curves.
