"""
Gaussian distributions and friends

TO DO:
    Improve communications between Gaussian variants. This is important when considering
    multi-phase training (e.g., start training mean, then covariance).
"""

from .BlockGauss import BlockDiagGauss, BlockDiagGaussMap
from .fixed_cov_gaussian_map import FactCovGaussianMap, FixedCovGaussianMap
from .Gauss import Gaussian, GaussianMap
from .gauss_hyperball import GaussHyperballMap
from .gauss_hypercube import GaussHypercubeMap
from .TGauss import (
    TensorizedGaussian,
    TensorizedGaussianMap,
    tgauss_to_gauss_param,
)
