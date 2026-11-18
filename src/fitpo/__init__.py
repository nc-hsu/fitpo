"""fitpo: fit piecewise linear approximations to a nonlinear pushover curve.

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
