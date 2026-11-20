# fitpo

`fitpo` fits piecewise-linear backbones to nonlinear pushover curves. A
pushover curve of a complex multi-degree-of-freedom structure is reduced to a
few straight segments (bilinear, trilinear or quadrilinear), which define the
backbone of an equivalent single-degree-of-freedom system. Depending on the
function the fitting procedure and optimisation objective can vary.

`fitpo` offers:

- the general fits of De Luca et al. (2013), with automatic choice between a
  trilinear and a quadrilinear shape;
- bilinear fits that end at the last point or at a set loss of strength e.g. 
  80% F_max;
- two fits for concentrically braced frames (CBFs), whose curves backbone curves
  don't follow the same elastic-hardening-softening-residual behaviour that the
  De Luca fits expect.;
- envelopes of cyclic pushover data, ready for the backbone fits.

## Which function do I need?

| Your data | Function |
|---|---|
| A monotonic pushover curve | {func}`fitpo.fit_deluca_backbone` (or {func}`fitpo.fit_bilinear_backbone` for a bilinear shape with a set end point) |
| A braced-frame curve with a brace-fracture drop, densely sampled | {func}`fitpo.fit_cbf_cusum_backbone` |
| A braced-frame curve with a fracture drop and a residual plateau, also sparse envelopes | {func}`fitpo.fit_cbf_knots_backbone` |
| A cyclic trace | {func}`fitpo.fit_envelope`, then one of the fits above on each envelope |

{func}`fitpo.area_error` scores any fit against its curve, so fits of
different shapes can be compared. See {doc}`method` for how each fit works.

```{admonition} Alpha software
:class: warning

fitpo is under active development, and its interface and results may change
between versions. If you need exactly reproducible results, pin the version,
e.g. `pip install fitpo==0.1.0`.
```

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

method
conventions
limitations
data
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/deluca_modes
examples/bilinear_modes
examples/cbf_cusum
examples/cbf_knots
examples/envelopes
examples/origin
examples/deluca_vs_cbf
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
references
citing
changelog
licence
```
