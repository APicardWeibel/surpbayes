"""
Distribution maps used in ADM1 related routines.

Main maps are
    proba_param_map (maps to all Gaussian distributions),
    proba_param_t_map (maps to Gaussian distributions with diagonal covariance)
    proba_param_fcov_map (maps to Gaussian distributions with covariance fixed to default)
"""

from .interface import Interface
from .standard_proba import prior_param, proba_map
