# Changelog

All notable changes to fitpo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The first public release, 0.1.0, will contain the following.

### Added

- `fit_deluca_backbone`: bilinear (elastic-perfectly-plastic), trilinear and
  quadrilinear backbones after De Luca et al. (2013), with `mode="auto"`
  choosing between trilinear and quadrilinear by a plateau-acceptance rule.
- `fit_bilinear_backbone`: bilinear backbones ending at the last point
  (`"bilinear-ultimate"`) or at a set loss of strength (`"bilinear-pctloss"`).
- `fit_cbf_cusum_backbone`: quadrilinear backbones for concentrically braced
  frames, with the brace-fracture drop found by CUSUM change detection.
- `fit_cbf_knots_backbone`: CBF backbones with the fracture knots placed at
  force fractions of the peak or at given displacements, ending on a residual
  plateau or dropping to zero force. Knot forces are solved exactly by linear
  programming for each trial yield force.
- `fit_envelope`: positive and negative envelopes of cyclic pushover data,
  with five tracing modes.
- `area_error`: the area between a curve and a fit over the whole curve,
  optionally normalised.
- `elastic_slope`: the secant elastic stiffness used by every fit.
- `FitResult`: a named tuple of `fitted_po`, `mode`, `objective` and
  `converged`, returned by every backbone function.
- `origin` option on every backbone function and `area_error`: refuse a curve
  that does not start at (0, 0) (default), shift its displacement or its
  whole first point, or use it as given.
- Input checks (shape, NaN/inf, decreasing displacement) and checks on every
  fitted shape, so an invalid backbone is never returned.
- Warnings when an optimiser does not converge, when a curve has no hardening
  branch, and when a CBF curve never reaches the residual force level.
- Type hints throughout (PEP 561 `py.typed`).

[Unreleased]: https://github.com/nc-hsu/fitpo/commits/main
