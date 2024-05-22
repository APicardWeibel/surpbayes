"""
Exponential family submodule.

Main classes:
    ExponentialFamily
Class for Exponential Family using the natural parametrisation. Initialization is modified to take
as input a sampling function, the feature map, normalisation function, reference log density and
derivatives of these functions. Kullback Leibler related methods are reimplemented. transform is
also reimplemented and output an ExponentialFamily object.

Future feature: subset method should be reimplemented to output an ExponentialFamily

    PreExpFamily
Class for Exponential Families not using the natural parametrisation. Initialization is modified to
take as input map to Proba objects, the feature map, the map to the natural parametrisation,
its inverse and derivative of the inverse. The transform method is reimplemented to output a
PreExpFamily object.


Objects:
    Beta: an instance of ExponentialFamily describing the beta family of distributions
    Exponential: an instance of ExponentialFamily describing the exponential family of
        distributions
    Pareto: an instance of ExponentialFamily describing the Pareto family of distributions
"""

from .beta import Beta
from .exponential import Exponential
from .exponential_family import ExponentialFamily
from .pareto import Pareto
from .pre_exponential_family import PreExpFamily
