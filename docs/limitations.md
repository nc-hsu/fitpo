# Limitations

fitpo is alpha software. Known limitations of the current version:

- **Monotonic curves only.** The backbone fits need a curve whose
  displacement never decreases and whose peak force is positive. Cyclic data
  must first be reduced to envelopes with {func}`fitpo.fit_envelope`, and the
  negative envelope negated.
- **A falling branch is needed** by the trilinear, quadrilinear and CBF fits.
  If the peak is the last data point they raise an error; use
  `bilinear-deluca` or {func}`fitpo.fit_bilinear_backbone` instead.
- **No hardening branch.** When the secant to the peak is at least as stiff
  as the elastic stiffness, the trilinear and quadrilinear De Luca fits warn
  and drop the hardening segment (see {doc}`method`).
- **The fits are near-optimal, not proven optimal.** Each search scans a
  coarse grid and then refines the best candidates, or solves an exact inner
  problem. On the test curves the De Luca fits are within 1 % of a dense
  brute-force search, but a curve with many local minima can still lead to a
  slightly worse local optimum.
- **CUSUM needs densely sampled curves.** On sparse envelopes,
  {func}`fitpo.fit_cbf_cusum_backbone` may detect no fracture and raise an
  error; use {func}`fitpo.fit_cbf_knots_backbone` instead.
- **Force-fraction knots depend on the curve crossing the levels.** In
  {func}`fitpo.fit_cbf_knots_backbone`, if the post-buckling branch stays
  above `joint_force_fraction` of the peak until the fracture, the joint knot
  lands at the bottom of the drop, and the post-buckling segment then has to
  span the drop too (its yield force can exceed the peak). If the curve never
  falls to `residual_force_fraction` of the peak, there is no plateau. Check
  the fit, and adjust the fractions or give the knots explicitly.
- **`bilinear-pctloss` ends before $d_{max}$** by design, at the chosen loss
  of strength.
- **Units are not checked.** Any consistent units work, but fitpo cannot tell
  if displacement and force are in mixed units.
