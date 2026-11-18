# Method

This page explains how each fit builds its backbone. All fits share two
ingredients: the area-error objective and the elastic stiffness rule.

## The area-error objective

A backbone $\hat F(d)$ is a piecewise-linear function of displacement. fitpo
measures how well it matches the pushover curve $F(d)$ by the area between
them,

$$
A = \int_{d_a}^{d_b} \left| F(d) - \hat F(d) \right| \, \mathrm{d}d ,
$$

and every free parameter of a fit is chosen to make this area as small as
possible. Both curves are piecewise linear, so fitpo integrates on a grid
that holds every data point and every breakpoint of the fit inside
$[d_a, d_b]$, using the trapezoid rule. Positive and negative differences
cannot cancel, because the absolute value is integrated.

### `res.objective` and `fitpo.area_error`

Each fit reports the area it minimised as `res.objective`. Its range
$[d_a, d_b]$ depends on the mode:

| Fit | Range of `res.objective` |
|---|---|
| `bilinear-deluca` | $[0, d_{max}]$ |
| `trilinear-deluca`, `quadrilinear-deluca` | $[d_y, d_{max}]$ |
| `bilinear-ultimate`, `bilinear-pctloss` | $[d_y, d_{end}]$ |
| `quadrilinear-cbf-cusum`, both `cbf-knots` modes | $[d_y, d_{max}]$ |

Because the ranges differ, `res.objective` should not be used to compare fits
of different modes. {func}`fitpo.area_error` integrates over the whole curve,
$[d_0, d_{max}]$, for any fit, and with `normalised=True` divides by
$\int |F| \, \mathrm{d}d$, which makes it dimensionless.

## The elastic stiffness

The elastic stiffness $k_e$ is the secant stiffness from the origin to the
point where the rising branch first reaches a fraction of the peak force,

$$
k_e = \frac{\alpha F_{max}}{d_{\alpha}} ,
$$

where $\alpha$ is `elastic_ref_fraction` (0.10 by default) and $d_\alpha$ is found by linear interpolation between the two samples
either side of that first crossing ({func}`fitpo.elastic_slope`). Taking the
first crossing keeps $k_e$ well defined when the force dips before the peak.
Every fit puts its yield point on the elastic line $F = k_e d$, so the yield
displacement follows from the yield force: $d_y = F_y / k_e$.

## The De Luca et al. fits

{func}`fitpo.fit_deluca_backbone` follows the rules of De Luca et al. (2013)
for fitting multilinear backbones to pushover curves. In short:

1. **Elastic segment.** The initial stiffness is the secant to 10 % of the
   peak base shear. The paper notes that the yield base shear could be used
   instead, but the peak gives the same accuracy without iterating, since the
   yield point is not known in advance.
2. **Hardening segment.** A segment with non-negative stiffness ends at the
   maximum base shear. Its slope is chosen to minimise the absolute area
   between the fit and the curve between the nominal yield point and the
   peak, where the yield point is the intersection with the elastic segment.
3. **Softening segment.** A negative-stiffness segment from the peak is also
   chosen by minimising the area over the softening region. A fourth,
   horizontal residual segment can be added: where steep softening turns
   distinctly milder partway down, the plateau belongs at that change; where
   the softening instead grows steeper, the plateau is kept only if it
   improves the fit by the area measure.

fitpo implements these rules in four modes. The physical meaning of each
segment is the usual one for a capacity curve: the elastic segment is the
undamaged structure; the hardening segment is the spread of yielding up to
the maximum strength; the softening segment is the loss of strength from
damage and second-order (P-delta) effects; and the plateau is the residual
strength the damaged structure keeps.

### `bilinear-deluca`

Elastic-perfectly-plastic: the elastic line up to the maximum force
$F_{max}$, then a horizontal line to the last data point,

$$
(0, 0) \;\to\; (F_{max}/k_e,\; F_{max}) \;\to\; (d_{max},\; F_{max}).
$$

Nothing is optimised. This shape also works when the peak is the last data
point.

### `trilinear-deluca`

Elastic, hardening and softening segments:

$$
(0, 0) \;\to\; (d_y, F_y) \;\to\; (d_{peak}, F_{max}) \;\to\; (d_{max}, F_{end}).
$$

The hardening slope $k_h$ is searched over $[0, F_{max}/d_{peak})$. Every
value in that range gives a yield point between the origin and the peak, at
the intersection of the hardening line through the peak with the elastic
line. The slope that minimises the area over $[d_y, d_{peak}]$ is kept. The
softening slope from the peak is then chosen to minimise the area over
$[d_{peak}, d_{max}]$.

### `quadrilinear-deluca`

As `trilinear-deluca`, but the softening segment ends at $(d_b, F_p)$ and is
followed by a horizontal plateau at $F_p$ to $d_{max}$:

$$
\dots \;\to\; (d_{peak}, F_{max}) \;\to\; (d_b, F_p) \;\to\; (d_{max}, F_p).
$$

$d_b$ and $F_p$ are searched together over $d_{peak} < d_b < d_{max}$ and
$0 < F_p < F_{max}$, so the shape always falls and then stays flat.

### `auto` and the plateau-acceptance rule

`mode="auto"` (the default) fits both shapes and compares their area errors
over the falling branch, $A_3$ (trilinear) and $A_4$ (quadrilinear). The
plateau has two more parameters, so it is kept only if all three of these
hold:

- it improves the fit clearly: $A_4 < (1 - 0.05)\,A_3$
  (`plateau_min_improvement`);
- its force is not negligible: $F_p \ge 0.10\,F_{max}$
  (`plateau_min_force_fraction`);
- it is not a short stub at the end: $d_{max} - d_b \ge 0.05\,(d_{max} -
  d_{peak})$ (`plateau_min_length_fraction`).

This implements the paper's "keep the plateau if it improves the fit"
criterion. The three thresholds are fitpo's defaults and can be changed.

### No hardening branch

A hardening line through the peak meets the elastic line before the peak
only if the secant from the origin to the peak, $F_{max}/d_{peak}$, is less
stiff than $k_e$. If it is not, as for a curve that is still nearly elastic
at its peak, no valid hardening segment exists. The trilinear and
quadrilinear modes then warn, set the yield point to
$(F_{max}/k_e, F_{max})$ (the limit of a flat hardening segment) and start the
softening segment there. The result has no separate peak row. For such
curves, `bilinear-deluca` or a CBF fit is usually the better choice.

## Bilinear fits

{func}`fitpo.fit_bilinear_backbone` fits an elastic segment and one
post-yield segment to a fixed end point on the curve:

$$
(0, 0) \;\to\; (d_y, F_y) \;\to\; (d_{end}, F_{end}).
$$

The yield force $F_y$ is chosen to minimise the area over $[d_y, d_{end}]$.
The two modes differ only in the end point:

- **`bilinear-ultimate`**: the last data point. If the curve ends below zero
  force, the end point is its last crossing of zero force instead, so the fit
  never runs into negative force.
- **`bilinear-pctloss`**: the first point after the peak where the force has
  fallen to `bilinear_end_fraction` of the peak (0.80 by default, i.e. a 20 %
  loss of strength), interpolated between samples. This is a common
  definition of the usable deformation capacity. If the force never falls
  that far, the last data point is used. The fit then ends at $d_{end}$, which
  can be well before $d_{max}$.

## Fits for concentrically braced frames

The pushover curve of a concentrically braced frame (CBF) typically has four
parts: an elastic rise; a gradual decline after the compression braces
buckle; a sudden drop when a brace fractures; and a lower residual branch
carried by the remaining braces and the frame. The De Luca shapes cannot
follow a sudden drop in the middle of the curve, so fitpo has two fits built
around it.

### CUSUM fracture detection

{func}`fitpo.fit_cbf_cusum_backbone` finds the fracture drop automatically
with a cumulative-sum (CUSUM) change detector (Page, 1954):

1. From the first sample at or beyond `cusum_search_start_fraction` of
   $d_{max}$ (0.25 by default), the slope $\mathrm{d}F/\mathrm{d}d$ of the
   curve is computed and standardised (mean removed, divided by its standard
   deviation over the search window).
2. Two running sums track sustained upward and downward departures of the
   standardised slope, each reduced by the drift $k$ (`cusum_drift`, 0.5) at
   every step; the downward sum is kept at or below zero:
   $S^-_i = \min\left(0,\; S^-_{i-1} + s_i + k\right)$.
3. Each time $S^-$ falls below $-h$ (`cusum_threshold`, 4.0), a sharp drop is
   recorded and the sum restarts. The fracture region runs from the sample
   before the first recorded drop to the sample after the last one.

The backbone is then fitted around that region:

- the **joint** (start of the drop) stays at the first fracture point's
  displacement, with its force free; the **post-buckling** segment runs from
  the yield point on the elastic line to the joint, and its slope and the
  joint force are chosen together to minimise the area of the post-buckling
  and drop segments;
- the **residual** segment starts at the last fracture point's displacement,
  with its force free, and runs to $d_{max}$ with a non-positive slope; its
  start force and slope are chosen to minimise its area;
- the **drop** joins the joint to the start of the residual segment.

CUSUM needs enough samples around the drop to see the change in slope, so it
suits densely sampled curves. On sparse envelopes it may detect no fracture,
and the function raises an error.

### Fixed knots

{func}`fitpo.fit_cbf_knots_backbone` fixes the displacements of the two
interior knots and fits all knot forces together. With the residual plateau
(the default) the shape is

$$
(0, 0) \;\to\; (d_y, F_y) \;\to\; (d_f, F_f) \;\to\; (d_r, F_r) \;\to\;
(d_{max}, F_r),
$$

i.e. elastic, post-buckling, fracture drop and horizontal residual plateau.
With `residual_plateau=False` the drop from $(d_f, F_f)$ continues straight
down to zero force at $d_{zero}$, with its own fitted slope, and the force
stays at zero to $d_{max}$ if $d_{zero} < d_{max}$.

The knot displacements are placed in one of two ways:

- **By force fraction** (default): $d_f$ is the first data point after the
  peak at or below `joint_force_fraction` $\times F_{max}$ (0.80), and $d_r$
  the first point after $d_f$ at or below `residual_force_fraction`
  $\times F_{max}$ (0.05).
- **By displacement**: `d_f` and `d_r` are given directly and need not be data
  points. This is the better choice when the knots are searched over (for
  example by an outer optimisation): snapping a fraction to a data point
  depends on how the curve is sampled, so adding points along a segment can
  move a knot and make the best fit worse, while explicit displacements
  cannot.

With force fractions, a meaningful quadrilinear fit needs the curve to cross
both levels after the peak. If it never falls to the joint level, the fit
raises an error. If it never falls to the residual level, $d_r$ becomes the
last data point, the plateau has zero length and the fit has only three
branches; a warning then suggests raising `residual_force_fraction`.

For a fixed yield force $F_y$, the backbone is a linear function of the
other knot forces. The area error is then a weighted sum of absolute
residuals, which has an exact solution: a small linear program with the
plateau, and a weighted median for the drop to zero. Only $F_y$, and for the
drop its zero-force point $d_{zero}$, need a 1-D search. This makes the fit
reliable despite the kinks of the absolute value, which defeat
gradient-based optimisers.

## Envelopes of cyclic data

{func}`fitpo.fit_envelope` turns a cyclic pushover trace into a positive and
a negative envelope, which can then be fitted like monotonic curves. The
modes differ in which points of the trace they keep: cycle peaks
(`peak-to-peak`), points beyond every earlier displacement (`loading-front`),
the extreme force in each displacement bin (`displacement-bins`), or the
peak-to-peak skeleton refined with loading-front points
(`adjusted-peak-to-peak`, `simplified-loading-front`). The last mode uses the
line simplification of Ramer (1972) and Douglas and Peucker (1973), but
measures the vertical (force) deviation from each chord rather than the
perpendicular distance, because displacement and force have different
units. The {doc}`examples/envelopes` page compares the modes.
