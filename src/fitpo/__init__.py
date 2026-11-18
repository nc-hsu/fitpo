"""fitpo: fit piecewise linear approximations to a nonlinear pushover curve.

A pushover curve of a complex multi-degree-of-freedom structure is reduced
to a few straight segments, the backbone of an equivalent single-degree-of-
freedom system. Every fit chooses its segments to minimise the area between
the backbone and the curve:

- :func:`fit_deluca_backbone`: bilinear, trilinear or quadrilinear fits
  after De Luca et al. (2013);
- :func:`fit_bilinear_backbone`: bilinear fits that end at the last point
  or at a set strength loss;
- :func:`fit_cbf_cusum_backbone` and :func:`fit_cbf_knots_backbone`: fits
  for concentrically braced frames, whose curves drop sharply when a brace
  fractures;
- :func:`fit_envelope`: positive and negative envelopes of cyclic data,
  ready for the backbone fits;
- :func:`area_error` and :func:`elastic_slope`: the measures the fits use.

This top-level package is fitpo's whole public interface; the underscore
modules behind it are private and may change without notice.
"""

from importlib.metadata import version

from fitpo._bilinear import fit_bilinear_backbone
from fitpo._cbf import fit_cbf_cusum_backbone, fit_cbf_knots_backbone
from fitpo._common import FitResult, area_error
from fitpo._deluca import elastic_slope, fit_deluca_backbone
from fitpo._envelope import fit_envelope

# Read from the installed package metadata so the version is written in one
# place only (pyproject.toml).
__version__ = version("fitpo")
__all__ = [
    "FitResult",
    "__version__",
    "area_error",
    "elastic_slope",
    "fit_bilinear_backbone",
    "fit_cbf_cusum_backbone",
    "fit_cbf_knots_backbone",
    "fit_deluca_backbone",
    "fit_envelope",
]
