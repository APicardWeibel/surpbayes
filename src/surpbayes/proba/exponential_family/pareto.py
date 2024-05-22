""" Pareto distributions with known minimum value 1"""

from typing import Callable

import numpy as np
from surpbayes.misc import prod
from surpbayes.proba._errors import RenormError
from surpbayes.proba.exponential_family.exponential_family import ExponentialFamily
from surpbayes.types import ProbaParam, Samples


def __T(x: Samples) -> np.ndarray:
    return np.log(x)


def valid_par(par: ProbaParam) -> None:
    if par[0] >= -1:
        raise RenormError("Pareto natural parameter should be <-1")


def __g(par: ProbaParam) -> float:
    valid_par(par)
    return -np.log(-1 - par[0])


def __der_g(par: ProbaParam) -> ProbaParam:
    valid_par(par)
    return -1 / (1 + par[0])


def __der_der_g(par: ProbaParam) -> np.ndarray:
    valid_par(par)
    return 1 / ((1 + par[0]) ** 2).reshape((1, 1))


def __gen(par: ProbaParam) -> Callable[[int], Samples]:
    valid_par(par)
    eta = -1 - par[0]

    def fun(n: int) -> Samples:
        return np.random.pareto(eta, (n, 1))

    return fun


def __h(x: Samples) -> np.ndarray:
    pre_shape = x.shape[:-1]
    x = x.flatten()
    out = np.zeros(prod(pre_shape))
    out[~np.apply_along_axis(lambda x: x[0] >= 1.0, -1, x)] = -np.inf
    return out.reshape(pre_shape)


Pareto = ExponentialFamily(
    gen=__gen,
    T=__T,
    g=__g,
    der_g=__der_g,
    der_der_g=__der_der_g,
    h=__h,
    proba_param_shape=(1,),
    sample_shape=(1,),
    ref_param=np.array([-2]),
)
