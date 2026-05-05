"""fitpo: fit piecewise linear approximations to a nonlinear pushover curve."""

from fitpo.cbf_fitting import fit_cbf_backbone
from fitpo.fitting import fit_piecewise_backbone

__version__ = "0.0.1"
__all__ = ["__version__", "fit_cbf_backbone", "fit_piecewise_backbone"]
