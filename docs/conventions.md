# Conventions

## Input curves

A pushover curve is an `(N, 2)` array (or anything numpy can turn into one):

- column 0: displacement, which must never decrease (repeated values are
  allowed, since analyses can record several steps at one displacement);
- column 1: force.

Any consistent units work: fitpo never converts units, and every result is in
the units of the input. The examples use mm and N.

All values must be finite; NaN or inf raise a `ValueError`. The backbone fits
need a positive peak force. To fit the negative side of a cyclic test, flip
the signs of its envelope first (see below).

## The `origin` option

Every backbone fit, and {func}`fitpo.area_error`, assumes the curve starts at
the unloaded point $(0, 0)$, because the elastic segment starts there. A
curve that starts elsewhere may either be missing its first, unloaded point, or
it may have an initial displacement due to the application of gravity loads. 
Fitpo does not guess a correction. The `origin` argument sets what happens:

| `origin` | Effect |
|---|---|
| `"strict"` (default) | Raise a `ValueError` that gives the first point and how far it is from $(0, 0)$, as a percentage of the largest displacement and force. |
| `"correct_disp"` | Subtract `d[0]` from every displacement. Forces are unchanged. |
| `"correct_dispforc"` | Subtract the whole first point `(d[0], f[0])`. |
| `"as-is"` | Use the curve as given. The elastic segment still starts at $(0, 0)$, so any initial offset counts as elastic. |

With a correction, `fitted_po` is in the corrected coordinates. In
{func}`fitpo.fit_cbf_knots_backbone`, explicit `d_f` and `d_r` are given in the
coordinates of the input curve and are shifted with it. Use the same `origin`
setting in {func}`fitpo.area_error` as in the fit, so the curve and the fit
are compared in the same coordinates. The {doc}`examples/origin` page shows
the effect of each setting.

## Envelopes of cyclic data

{func}`fitpo.fit_envelope` returns `(positive_envelope, negative_envelope)`.
Both start at `[0, 0]`. The positive envelope has positive displacements and
forces; the **negative envelope keeps the signs of the data**, so its
displacements and forces are negative. To fit a backbone to the negative
side, negate it first:

```python
pos, neg = fitpo.fit_envelope(cyclic)
res_neg = fitpo.fit_deluca_backbone(-neg)  # back in positive coordinates
```

## Results: `FitResult`

Every backbone function returns a {class}`fitpo.FitResult`, a named tuple
with four fields:

| Field | Meaning |
|---|---|
| `fitted_po` | `(M, 2)` array of the backbone's breakpoints, one `[d, F]` row each, starting at `[0, 0]`. |
| `mode` | The shape actually fitted, e.g. `"trilinear-deluca"`. With `mode="auto"` this says which shape was chosen. |
| `objective` | The value the fit minimised, over a range that depends on the mode (see {doc}`method`). |
| `converged` | `True` if every optimiser call reported success. If not, the fit also warns. |

Every returned `fitted_po` is checked: its values are finite, its first row
is `[0, 0]`, the yield displacement is positive and the displacements
strictly increase from row to row. A segment of zero length is never
returned.

## Rows of `fitted_po`

| Mode | Rows |
|---|---|
| `bilinear-deluca` | origin, yield $(F_{max}/k_e, F_{max})$, end $(d_{max}, F_{max})$ |
| `trilinear-deluca` | origin, yield, peak, end |
| `quadrilinear-deluca` | origin, yield, peak, plateau start, end |
| `bilinear-ultimate`, `bilinear-pctloss` | origin, yield, end point |
| `quadrilinear-cbf-cusum` | origin, yield, joint, residual start, end |
| `quadrilinear-cbf-knots` | origin, yield, joint $(d_f, F_f)$, residual start $(d_r, F_r)$, end $(d_{max}, F_r)$ |
| `trilinear-cbf-knots` | origin, yield, joint, zero-force point $(d_{zero}, 0)$, and $(d_{max}, 0)$ if $d_{zero} < d_{max}$ |

The number of rows can be smaller than in this table: without a hardening
branch the De Luca fits have no separate peak row, and the knots fit drops
its last row when $d_r = d_{max}$. **Always use `len(res.fitted_po)`** rather
than assuming a row count, and check `res.mode` for the shape.
