"""fitpo: fit piecewise linear approximations to a nonlinear pushover curve."""

from fitpo.cbf_fitting import fit_cbf_backbone, fit_tetralinear_backbone
from fitpo.envelope import fit_envelope
from fitpo.fitting import elastic_slope, fit_piecewise_backbone

__version__ = "0.0.1"
__all__ = [
    "__version__",
    "elastic_slope",
    "fit_cbf_backbone",
    "fit_envelope",
    "fit_piecewise_backbone",
    "fit_tetralinear_backbone",
]
