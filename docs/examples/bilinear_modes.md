---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Bilinear fits

{func}`fitpo.fit_bilinear_backbone` fits an elastic segment and one
post-yield segment. `"bilinear-ultimate"` ends at the last data point;
`"bilinear-pctloss"` ends where the force has fallen to
`bilinear_end_fraction` of the peak (0.80 by default, a 20 % loss of
strength). The curves are envelopes from the example data (see
{doc}`../data`).

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
names = ["envelope_3s_cbf_dc2_10", "envelope_3s_cbf_dc2_41", "envelope_3s_cbf_dc2_71"]
curves = {n: np.loadtxt(DATA / "bilinear" / f"{n}.csv", delimiter=",") for n in names}
```

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="w")
for ax, (name, curve) in zip(axes, curves.items()):
    ax.plot(curve[:, 0], curve[:, 1] / 1e3, color="0.6", lw=3, label="curve")
    for mode, style in (("bilinear-ultimate", "o-"), ("bilinear-pctloss", "s--")):
        res = fitpo.fit_bilinear_backbone(curve, mode=mode)
        ax.plot(res.fitted_po[:, 0], res.fitted_po[:, 1] / 1e3, style, label=mode)
    peak = curve[:, 1].max()
    ax.axhline(0.8 * peak / 1e3, color="tab:orange", ls=":", lw=1, label="0.80 peak")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("roof displacement [mm]")
axes[0].set_ylabel("base shear [kN]")
axes[0].legend(fontsize=8)
fig.tight_layout()
```

`"bilinear-pctloss"` stops at the 20 % strength-loss point, so its yield
force is fitted to the curve up to that point only. `"bilinear-ultimate"`
averages over the whole curve, including any sharp drop at the end.
