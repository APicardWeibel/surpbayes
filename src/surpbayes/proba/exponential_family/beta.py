r"""
Beta distributions

Density is
..math::
    \exp(
        (\alpha - 1)\log(x) + (\beta - 1)\log(1-x)
        + \log(\Gamma(\alpha))) + \log(\Gamma(\beta))
        - \log(\Gamma(\alpha + \beta))
        )

for x in [0,1], 0 if not.

"""

from typing import Callable

import numpy as np
from scipy.special import digamma, gamma, polygamma
from surpbayes.proba._errors import RenormError
from surpbayes.proba._helper import prod
from surpbayes.proba.exponential_family.exponential_family import \
    ExponentialFamily
from surpbayes.types import ProbaParam, Samples


def __T(x: Samples) -> ProbaParam:
    pre_shape = x.shape[:-1]
    out = np.zeros((prod(pre_shape), 2))
    x = x.flatten()
    good_index = (x > 0) & (x < 1)

    out[good_index] = np.array([np.log(x[good_index]), np.log(1 - x[good_index])]).T
    return out.reshape(pre_shape + (2,))


def __g(par: ProbaParam) -> float:
    if np.any(par <= -1):
        raise RenormError(f"{par} must be positive")
    return np.sum(np.log(gamma(par + 1))) - np.log(gamma(np.sum(par + 1)))


def __der_g(par: ProbaParam) -> ProbaParam:
    return digamma(par + 1) - digamma(np.sum(par + 1))


def __der_der_g(par: ProbaParam) -> np.ndarray:
    return np.diag(polygamma(1, par + 1)) - polygamma(1, np.sum(par + 1))


def __h(x: Samples) -> np.ndarray:
    pre_shape = x.shape[:-1]
    x = x.flatten()
    out = np.zeros(prod(pre_shape))
    out[~((x < 1.0) & (x > 0.0))] = -np.inf
    return out.reshape(pre_shape)


def __gen(par: ProbaParam) -> Callable[[int], Samples]:
    def fun(n: int) -> Samples:
        return np.random.beta(par[0] + 1, par[1] + 1, size=n).reshape((n, 1))

    return fun


Beta = ExponentialFamily(
    gen=__gen,
    T=__T,
    g=__g,
    der_g=__der_g,
    der_der_g=__der_der_g,
    h=__h,
    proba_param_shape=(2,),
    sample_shape=(1,),
    ref_param=np.array([0.0, 0.0]),
    np_out=True,
)
