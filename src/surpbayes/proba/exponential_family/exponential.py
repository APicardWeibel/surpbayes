""" Exponential distributions """

from typing import Callable

import numpy as np
from surpbayes.misc import prod
from surpbayes.proba._errors import RenormError
from surpbayes.proba.exponential_family.exponential_family import ExponentialFamily
from surpbayes.types import ProbaParam, Samples


def valid_par(par: ProbaParam) -> None:
    if par[0] <= 0:
        raise RenormError("Pareto natural parameter should be <-1")


def __T(x: Samples) -> np.ndarray:
    return -x


def __g(par: ProbaParam) -> float:
    valid_par(par)
    return -np.log(par[0])


def __der_g(par: ProbaParam) -> ProbaParam:
    valid_par(par)
    return -1 / par


def __der_der_g(par: ProbaParam) -> np.ndarray:
    valid_par(par)
    return (1 / (par**2)).reshape((1, 1))


def __gen(par: ProbaParam) -> Callable[[int], Samples]:
    def fun(n: int) -> Samples:
        return np.random.exponential(1 / par[0], (n, 1))

    return fun


def __h(xs: Samples) -> np.ndarray:
    pre_shape = xs.shape[:-1]
    xs = xs.flatten()
    out = np.zeros(prod(pre_shape))
    out[~np.apply_along_axis(lambda x: x[0] > 0.0, -1, xs)] = -np.inf
    return out.reshape(pre_shape)


def __der_h(xs: Samples) -> np.ndarray:
    pre_shape = xs.shape[:-1]
    xs = xs.flatten()
    out = np.zeros(prod(pre_shape))
    out[~np.apply_along_axis(lambda x: x[0] > 0.0, -1, xs)] = -np.inf
    return out.reshape(pre_shape + (1,))


Exponential = ExponentialFamily(
    gen=__gen,
    T=__T,
    g=__g,
    der_g=__der_g,
    der_der_g=__der_der_g,
    h=__h,
    der_h=__der_h,
    proba_param_shape=(1,),
    sample_shape=(1,),
    ref_param=np.array([1]),
    np_out=True,
)
