# fitpo

<!-- Zenodo DOI badge: added after the first release. -->

fitpo fits piecewise-linear backbones to nonlinear pushover curves. A
pushover curve of a complex multi-degree-of-freedom (MDOF) structure is
reduced to a few straight segments — bilinear, trilinear or quadrilinear —
which define the backbone of an equivalent single-degree-of-freedom (SDOF)
system. Every fit chooses its segments to minimise the area between the
backbone and the curve. Besides the general fits after De Luca et al. (2013),
fitpo has fits for concentrically braced frames (CBFs), whose curves drop
sharply when a brace fractures, and a tool to extract envelopes from cyclic
pushover data.

> **Alpha software.** fitpo is under active development and its interface
> and results may change between versions. If you need exactly reproducible
> results, pin the version (e.g. `fitpo==0.1.0`).

## Installation

```bash
pip install fitpo
pip install "fitpo[plot]"   # with matplotlib, for plotting
```

fitpo needs Python 3.12 or later.

## Quick start

```python
import numpy as np
import fitpo

# A pushover curve: roof displacement [mm], base shear [N], starting at (0, 0).
d = np.linspace(0.0, 120.0, 241)
f = np.interp(d, [0, 10, 40, 80, 120], [0, 1000, 1200, 600, 600])
curve = np.column_stack([d, f])

res = fitpo.fit_deluca_backbone(curve)  # "auto": trilinear or quadrilinear
print(res.mode)  # quadrilinear-deluca
print(res.fitted_po.round(1))  # breakpoints [d, F] of the backbone
print(round(fitpo.area_error(curve, res.fitted_po, normalised=True), 4))  # 0.0
```

Any consistent units work. Every backbone function returns a `FitResult` with
the breakpoints (`fitted_po`), the shape fitted (`mode`), the area error
(`objective`) and whether the optimisers converged (`converged`).

## Which function?

| Function | Use it for |
|---|---|
| `fit_deluca_backbone` | General pushover curves: bilinear, trilinear or quadrilinear backbones after De Luca et al. (2013); `mode="auto"` picks trilinear or quadrilinear. |
| `fit_bilinear_backbone` | A bilinear backbone that ends at the last point (`"bilinear-ultimate"`) or at a set loss of strength (`"bilinear-pctloss"`). |
| `fit_cbf_cusum_backbone` | Densely sampled CBF curves: finds the brace-fracture drop automatically (CUSUM change detection). |
| `fit_cbf_knots_backbone` | CBF curves, including sparse envelopes: places the fracture knots at force fractions of the peak or at given displacements. |
| `fit_envelope` | Cyclic pushover data: extracts positive and negative envelopes, ready for the backbone fits. |

`area_error` compares any fit with a curve over the whole displacement range.

## Documentation

- Documentation: <https://fitpo.readthedocs.io>
- Source code and issues: <https://github.com/nc-hsu/fitpo>

## How to cite

If you use fitpo in your work, please cite the software; the citation details
are in
[`CITATION.cff`](https://github.com/nc-hsu/fitpo/blob/main/CITATION.cff)
(GitHub's "Cite this repository" button gives them as APA or BibTeX). If you
use `fit_deluca_backbone`, please also cite the method:

> De Luca, F., Vamvatsikos, D. and Iervolino, I. (2013). Near-optimal
> piecewise linear fits of static pushover capacity curves for equivalent
> SDOF analysis. *Earthquake Engineering & Structural Dynamics*, 42(4),
> 523–543. <https://doi.org/10.1002/eqe.2225>

## Licence

Licensed under GPL-3.0-or-later; see
[LICENSE](https://github.com/nc-hsu/fitpo/blob/main/LICENSE).

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details.
