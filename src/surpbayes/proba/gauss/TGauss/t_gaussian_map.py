"""
Map for Gaussian distributions with diagonal covariance.
"""

import warnings
from math import log, pi
from typing import Callable, Optional

import numpy as np
from surpbayes.misc import _get_pre_shape, prod
from surpbayes.proba._errors import RenormError
from surpbayes.proba._helper import _shape_info
from surpbayes.proba.exponential_family import PreExpFamily
from surpbayes.proba.gauss.TGauss.t_gaussian import TensorizedGaussian
from surpbayes.types import ProbaParam, Samples


def exp_family_tgauss(
    sample_size: Optional[int] = None,
    sample_shape: Optional[tuple[int, ...]] = None,
):
    r"""Prepare functions to interpret block diagonal gaussian distributions as exponential family

    Compute the T for exponential family interpretation of Gaussian distributions.

    Should be coherent with versions implemented for Gaussian distributions.
    """

    sample_size, sample_shape = _shape_info(sample_size, sample_shape)

    t_shape = sample_shape[:-1] + (2 * sample_shape[-1],)

    def T(xs: Samples) -> np.ndarray:
        return np.concatenate([xs, -0.5 * xs**2], -1)

    def param_to_T(param: ProbaParam) -> np.ndarray:
        mean = param[0]
        half_var = param[1]
        inv_var = half_var ** (-2)
        trans_mean = inv_var * mean
        out = np.zeros(2 * sample_size)
        out[:sample_size] = trans_mean
        out[sample_size:] = inv_var
        return out

    # def T_to_quadra(t_val):
    def T_to_param(t_val: np.ndarray) -> ProbaParam:
        H = t_val[sample_size:]
        # Compute center
        center = (1 / H) * t_val[:sample_size]

        if np.min(H) < 0:
            raise RenormError("Inverse variance must be positive")

        half_var = np.sqrt(1 / H)
        accu = np.zeros((2, sample_size))  # type: ignore
        accu[0] = center
        accu[1] = half_var
        return accu

    def der_T_to_param(t_val: np.ndarray) -> np.ndarray:
        out = np.zeros((2 * sample_size, 2 * sample_size))  # type: ignore
        x = t_val[sample_size:]
        H = t_val[sample_size:]

        if np.min(H) < 0:
            raise RenormError("Inverse variance must be positive")

        out[:sample_size, :sample_size] = np.diag(1 / H)
        out[:sample_size, sample_size:] = np.diag(-x / H**2)
        out[sample_size:, sample_size:] = -1 / 2 * np.diag(H ** (3 / 2))

        return out

    def g(t_val: np.ndarray):
        return 0.5 * (
            sample_size * log(2 * pi)
            - np.sum(np.log(t_val[sample_size:]))
            + np.sum(t_val[:sample_size] ** 2 / t_val[sample_size:])
        )

    def der_g(t_val):
        out = np.zeros(t_shape)
        inv_M = t_val[sample_size:] ** (-1)
        out[:sample_size] = inv_M * t_val[:sample_size]
        out[sample_size:] = -0.5 * inv_M - 0.5 * out[:sample_size] ** 2
        return out

    return T, param_to_T, T_to_param, der_T_to_param, g, der_g, t_shape


class TensorizedGaussianMap(PreExpFamily):
    """
    For Gaussian tensorized distributions, use the following subclass (kl and grad_kl are overriden)
    The distribution param shape is (2, sample_shape)
    The first element is the mean, the second element controls the standard deviation through:
    sigma = np.abs(x)
    Note that this class could be reimplemented using the reparametrize from GaussianMap.
    This implementation is slightly more efficient as it takes advantage of the fact that the
    covriance is diagonal.
    """

    map_type = "Gaussian"

    def __init__(
        self,
        sample_size: Optional[int] = None,
        sample_shape: Optional[tuple[int, ...]] = None,
    ):
        """
        Construct the family of gaussian distributions with independant components
        (tensorized gaussians), from the shape of the sample.

        Either sample_size or sample_shape must be specified. If both are, sample_size is ignored.

        The resulting family is parametrized by objects of shape  (2, sample_shape),
        the first element being the mean, the second element controling the standard deviation through:
                sigma = np.abs(x)
        """

        sample_size, sample_shape = _shape_info(sample_size, sample_shape)

        proba_param_shape = (2, sample_size)

        def prob_map(x: ProbaParam) -> TensorizedGaussian:
            x = np.array(x)
            if x.shape != proba_param_shape:
                warnings.warn(
                    "\n".join(
                        [
                            f"Distribution parameter shape is {x.shape} (Expected{proba_param_shape})",
                            "Trying to construct nonetheless. Watch out for strange behaviour.",
                        ]
                    )
                )

            return TensorizedGaussian(
                means=x[0], devs=np.abs(x[1]), sample_shape=sample_shape
            )

        def log_dens_der(x: ProbaParam) -> Callable[[Samples], np.ndarray]:
            x = np.array(x)
            means = x[0]
            signed_devs = x[1]
            inv_var = signed_devs ** (-2)

            def derivate(samples: Samples) -> np.ndarray:
                pre_shape = _get_pre_shape(samples, sample_shape)  # type: ignore
                der = np.zeros((prod(pre_shape),) + proba_param_shape)
                centered = samples.reshape((prod(pre_shape), sample_size)) - means  # type: ignore
                der[:, 0] = centered * inv_var
                der[:, 1] = -(1 - (centered**2) * inv_var) / signed_devs
                return der.reshape(pre_shape + proba_param_shape)

            return derivate

        ref_param = np.zeros(proba_param_shape)
        ref_param[1] = np.ones(sample_size)

        (
            T,
            param_to_T,
            T_to_param,
            der_T_to_param,
            g,
            der_g,
            t_shape,
        ) = exp_family_tgauss(sample_size, sample_shape)

        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            T=T,
            param_to_T=param_to_T,
            T_to_param=T_to_param,
            der_T_to_param=der_T_to_param,
            g=g,
            der_g=der_g,
            ref_param=ref_param,
            proba_param_shape=proba_param_shape,
            sample_shape=sample_shape,
            t_shape=t_shape,
        )

    def kl(
        self,
        param_1: ProbaParam,
        param_0: ProbaParam,
        n_sample: int = 0,
    ) -> float:
        """
        Computes the Kullback Leibler divergence between two tensorized gaussian distributions
        defined by their distribution parameters.

        Args:
            proba_1, proba_0 are 2 distribution parameters
            n_sample is disregarded

        Output:
            kl(proba_1, proba_0)
        """
        dim = self.sample_size

        means1, means0 = param_1[0], param_0[0]
        vars1, vars0 = param_1[1] ** 2, param_0[1] ** 2

        diff_mean = means1 - means0

        return 0.5 * (
            np.sum(np.log(vars0) - np.log(vars1))
            - dim
            + np.sum(vars1 / vars0)
            + np.sum((vars0 ** (-1)) * (diff_mean**2))
        )

    def grad_kl(
        self, param_0: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        mean0, vars0 = param_0[0], param_0[1] ** 2

        def fun(
            param_1: ProbaParam,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> tuple[np.ndarray, float]:
            der = np.zeros(self.proba_param_shape)
            vars1 = param_1[1] ** 2

            der[0] = (param_1[0] - mean0) * (vars0 ** (-1))
            der[1] = param_1[1] * (vars0 ** (-1) - vars1 ** (-1))

            return der, self.kl(param_1, param_0)

        return fun

    def grad_right_kl(
        self, param_1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        mean1, vars1 = param_1[0], param_1[1] ** 2

        def fun(
            param_0: np.ndarray,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> tuple[np.ndarray, float]:
            der = np.zeros(self.proba_param_shape)
            vars0 = param_0[1] ** 2
            diff_mean = param_0[0] - mean1
            der[0] = diff_mean * (vars0 ** (-1))
            der[1] = (-(diff_mean**2) / vars0 + 1 - (vars1 / vars0)) / param_0[1]

            return der, self.kl(param_1, param_0)

        return fun

    def to_param(self, g_proba: TensorizedGaussian) -> ProbaParam:
        """
        Transforms a Tensorized Gaussian back to a np.ndarray such that
        self(self.to_param(g_proba)) = g_proba
        """
        accu = np.zeros(self.proba_param_shape)
        accu[0] = g_proba.means
        accu[1] = g_proba.devs

        return accu


def tgauss_to_gauss_param(prob_param: ProbaParam) -> ProbaParam:
    """Convert a ProbaParam for TensorizedGaussianMap to a ProbaParam for ProbaMap resulting in
    the same gaussian distribution
    """
    prob_param = np.array(prob_param)
    proba_param_shape = prob_param.shape
    if len(proba_param_shape) != 2:
        raise ValueError(
            "A ProbaParam for TensorizedGaussianMap should be 2 dimensional"
        )

    if proba_param_shape[0] != 2:
        raise ValueError(
            "A ProbaParam for TensorizedGaussianMap should be shaped '(2, n)'"
        )

    accu = np.zeros((proba_param_shape[1] + 1, proba_param_shape[1]))
    accu[0] = prob_param[0]  # Passing means
    accu[1:] = np.diag(prob_param[1])  # Setting standard deviations

    return accu
