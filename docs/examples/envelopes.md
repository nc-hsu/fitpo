---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Envelopes of cyclic data

{func}`fitpo.fit_envelope` reduces a cyclic pushover trace to a positive and
a negative envelope. Its five modes differ in which points of the trace they
keep (see {doc}`../method`). This page compares them on the cyclic curve from
the example data (see {doc}`../data`) and on two synthetic traces where the
modes differ most.

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
MODES = [
    "peak-to-peak",
    "loading-front",
    "displacement-bins",
    "adjusted-peak-to-peak",
    "simplified-loading-front",
]


def plot_modes(trace, title, force_scale=1.0, unit="", **kwargs):
    """Plot the trace and the envelopes of every mode, one panel per mode."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharey=True, facecolor="w")
    for ax, mode in zip(axes, MODES):
        pos, neg = fitpo.fit_envelope(trace, mode=mode, **kwargs)
        ax.plot(trace[:, 0], trace[:, 1] / force_scale, color="0.8", lw=1)
        for env in (pos, neg):
            ax.plot(env[:, 0], env[:, 1] / force_scale, ".-", color="tab:red", lw=1.5)
        ax.set_title(mode, fontsize=10)
        ax.set_xlabel("displacement")
    axes[0].set_ylabel(f"force {unit}")
    fig.suptitle(title)
    fig.tight_layout()
```

## The example cyclic curve

```{code-cell} python
cyclic = np.loadtxt(DATA / "cyclic" / "po_curve_1.csv", delimiter=",")
plot_modes(cyclic, "examples/data/cyclic/po_curve_1.csv", force_scale=1e3, unit="[kN]")
```

## A fracture between cycle peaks

A synthetic trace whose loading backbone peaks at 250 between two cycle tips,
then drops sharply (a fracture). `peak-to-peak` only joins the cycle tips, so
it misses the peak; the loading-front modes capture it.

```{code-cell} python
def fracture_cyclic():
    """Growing cycles on a backbone that peaks at d = 3.2 and fractures by d = 4."""
    knots_d = [0.0, 1.0, 2.0, 3.0, 3.2, 4.0, 5.0]
    knots_f = [0.0, 100.0, 180.0, 240.0, 250.0, 80.0, 75.0]
    d_parts, f_parts, cur = [[0.0]], [[0.0]], 0.0
    targets = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, 5.0]
    for k, target in enumerate(targets):
        seg = np.linspace(cur, target, 60)[1:]
        d_parts.append(seg)
        f_parts.append(np.sign(target) * np.interp(np.abs(seg), knots_d, knots_f))
        cur = target
        if k != len(targets) - 1:  # unload to zero, except after the last push
            back = np.linspace(target, 0.0, 60)[1:]
            peak_f = np.sign(target) * np.interp(abs(target), knots_d, knots_f)
            d_parts.append(back)
            f_parts.append(peak_f * back / target)
            cur = 0.0
    return np.column_stack([np.concatenate(d_parts), np.concatenate(f_parts)])


plot_modes(fracture_cyclic(), "fracture between cycle peaks", include_final_peak=True)
```

## A drop below the peak-to-peak chord

A high cycle reaches force 200 at displacement 2 before fracture; a later,
degraded cycle reaches displacement 4 at only force 55. The loading front
between them lies far below the chord joining the two tips. `peak-to-peak`,
`displacement-bins` (whose bins hold only the two tips here) and
`adjusted-peak-to-peak` (which only moves the envelope outward) follow the
chord and so overestimate the falling branch; `loading-front` and
`simplified-loading-front` follow the drop.

```{code-cell} python
def postpeak_drop_cyclic():
    """A cycle to (2, 200), then a degraded cycle to (4, 55), mirrored."""
    d_parts, f_parts = [[0.0]], [[0.0]]
    for target, peak_f in [(2.0, 200.0), (-2.0, -200.0), (4.0, 55.0), (-4.0, -55.0)]:
        for seg in (np.linspace(0.0, target, 60)[1:], np.linspace(target, 0.0, 60)[1:]):
            d_parts.append(seg)
            f_parts.append(peak_f * seg / target)
    return np.column_stack([np.concatenate(d_parts), np.concatenate(f_parts)])


plot_modes(postpeak_drop_cyclic(), "drop below the peak-to-peak chord")
```

## Fitting an envelope

The envelopes can be fitted like monotonic curves. The negative envelope
keeps the signs of the data, so it is negated before fitting:

```{code-cell} python
pos, neg = fitpo.fit_envelope(cyclic, mode="simplified-loading-front")
for side, env in (("positive", pos), ("negative", -neg)):
    res = fitpo.fit_bilinear_backbone(env)
    print(side, res.mode, res.fitted_po.round(1).tolist())
```
