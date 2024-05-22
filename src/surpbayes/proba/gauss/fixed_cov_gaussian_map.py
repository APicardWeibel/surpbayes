"""
Special GaussianMap when the covariance is either fixed or fixed up to a factor.

Rationale for the classes:
    - Standard gaussian map heavily relies on covariance matrix construction/inversion
    when computing KL related quantiites. In this setting, this information can be predetermined.
"""

from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike
from surpbayes.misc import _get_pre_shape, check_shape, prod
from surpbayes.proba._errors import RenormError
from surpbayes.proba._helper import _shape_info
from surpbayes.proba.exponential_family.pre_exponential_family import PreExpFamily
from surpbayes.proba.gauss.Gauss import Gaussian, inverse_cov
from surpbayes.types import ProbaParam, Samples


def exp_family_gauss_fixed_cov(
    cov: np.ndarray,
    sample_size: Optional[int] = None,
    sample_shape: Optional[tuple[int, ...]] = None,
):
    r"""Prepare functions to interpret fixed cov gaussian distributions as exponential family

    The output T works for the following parametrisation of a gaussian distribution:
        $(Cov @ mean, (Cov_{i,i})_i, (Cov_{i,j})_{i>j})$.

    -.5 * (mean - x) Cov (mean - x)
    Cov @ mean, x
    """

    sample_size, sample_shape = _shape_info(sample_size, sample_shape)

    if not cov.shape == (sample_size, sample_size):
        raise ValueError(
            " ".join(
                [
                    "cov shape incoherent with sample_size, sample_shape information passed",
                    f"(expected {(sample_size, sample_size)}, got ({cov.shape}))",
                ]
            )
        )
    inv_cov = inverse_cov(cov)

    t_dim = sample_size

    def T(samples: Samples) -> np.ndarray:
        pre_shape = _get_pre_shape(samples, exp_shape=sample_shape)  # type: ignore
        return samples.reshape(pre_shape + (sample_size,))  # type: ignore

    def param_to_T(param: ProbaParam) -> np.ndarray:
        # parametrised by the mean
        return inv_cov @ param

    # def T_to_quadra(t_val):
    def T_to_param(t_val: np.ndarray) -> ProbaParam:
        return cov @ t_val

    def g(t_val):
        # Removed constants
        return 0.5 * (np.sum(t_val * (cov @ t_val)))

    def der_g(t_val):
        return cov @ t_val

    return T, param_to_T, T_to_param, g, der_g, (t_dim,)


class FixedCovGaussianMap(PreExpFamily):
    r"""
    Class for Gaussian probability distributions of form $\mathcal{N}(\mu, \Sigma)$ with $\Sigma$
    fixed.
    """

    # Indicate that this map deals with Gaussian
    map_type = "Gaussian"

    def __init__(
        self,
        sample_size: Optional[int] = None,
        sample_shape: Optional[tuple[int, ...]] = None,
        cov: Optional[ArrayLike] = None,
    ):
        sample_size, sample_shape = _shape_info(sample_size, sample_shape)

        # Set up default covariance if cov is None
        if cov is None:
            cov = np.eye(sample_size)
        else:
            cov = np.array(cov)
            check_shape(cov, (sample_size, sample_size))

        T, param_to_T, T_to_param, g, der_g, (t_dim,) = exp_family_gauss_fixed_cov(
            cov, sample_size, sample_shape
        )

        inv_cov = inverse_cov(cov)  # pre compute it once and for all

        def prob_map(x: ProbaParam) -> Gaussian:
            x = np.array(x)  # Force convert to array
            check_shape(x, (sample_size,))  # type: ignore

            return Gaussian(means=x, cov=cov, sample_shape=sample_shape)  # type: ignore

        def log_dens_der(x: ProbaParam) -> Callable[[Samples], np.ndarray]:
            means = x

            def derivative(samples: Samples) -> np.ndarray:
                pre_shape = _get_pre_shape(samples, sample_shape)  # type: ignore
                centered = samples.reshape((pre_shape) + (sample_size,)) - means  # type: ignore
                return centered @ inv_cov

            return derivative

        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            T=T,
            param_to_T=param_to_T,
            T_to_param=T_to_param,
            g=g,
            der_g=der_g,
            t_shape=(t_dim,),
            ref_param=np.zeros(sample_size),
            proba_param_shape=(sample_size,),
            sample_shape=sample_shape,
        )

        self._cov = cov
        self._inv_cov = inv_cov

    @property
    def cov(self):
        return self._cov

    @property
    def inv_cov(self):
        return self._inv_cov

    def __repr__(self) -> str:
        return str.join(
            "\n",
            [
                f"Gaussian Prior Map with fixed covariance on arrays of shape {self.sample_shape}.",
                f"Covariance:\n{self.cov}",
            ],
        )

    def kl(self, param_1: np.ndarray, param_0: np.ndarray, n_sample: int = 0) -> float:
        """Computes the Kullback Leibler divergence between two gaussian distributions.
        defined by their prior parameters.

        Args:
            proba_1, proba_0 are 2 prior parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            kl(proba_1, proba_0)
        """

        diff_means = param_0 - param_1
        kl = np.dot(diff_means, self.inv_cov @ diff_means)
        kl = kl / 2
        return kl

    def grad_kl(
        self, param_0: np.ndarray
    ) -> Callable[[np.ndarray, int], tuple[ProbaParam, float]]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        param_0 ->(param_1  ->  (nabla_{param_1} kl(param_1, param_0)))

        Args:
            param_0 is a parameter describing the right distribution.
        Output:
            nabla_{param_1}kl(param_1, param_0)
            See doc for more information.
        """

        inv_cov = self.inv_cov

        def fun(
            param_1: np.ndarray,
            n_sample: int = 0,  # pylint: disable=W0613
        ):
            diff_mean = param_1 - param_0
            grad_kl = inv_cov @ diff_mean
            return grad_kl, 0.5 * np.sum(diff_mean * grad_kl)

        return fun

    def grad_right_kl(
        self, param_1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the second distribution
        (nabla_{param_0} kl(param_1, param_0))

        Args:
            param_1, param_0 are 2 distribution parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            nabla_{param_1}kl(param_1, param_0)
            See doc for more information.
        """
        inv_cov = self.inv_cov

        def fun(
            param_0: ProbaParam,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> tuple[ProbaParam, float]:
            diff_mean = param_0 - param_1
            grad_kl = inv_cov @ diff_mean
            return grad_kl, 0.5 * np.sum(diff_mean * grad_kl)

        return fun

    def to_param(self, g_proba: Gaussian) -> np.ndarray:
        """
        Transforms a Gaussian back to a np.ndarray such that
        self(self.to_param(g_proba)) = g_proba
        """
        return g_proba.means


def exp_family_gauss_fact_cov(
    cov: np.ndarray,
    sample_size: Optional[int] = None,
    sample_shape: Optional[tuple[int, ...]] = None,
):
    r"""Prepare functions to interpret fact cov gaussian distributions as exponential family

    The output T works for the following parametrisation of a gaussian distribution:
        $(inv_Cov @ mean, \sigma^-2)$.

        -.5 * inv_Cov (x x^T), \sigma^-2
    """

    sample_size, sample_shape = _shape_info(sample_size, sample_shape)

    assert cov.shape == (sample_size, sample_size)
    inv_cov = inverse_cov(cov)

    t_dim = sample_size + 1

    def T(samples: Samples) -> np.ndarray:
        pre_shape = _get_pre_shape(samples, exp_shape=sample_shape)  # type: ignore
        n_element = prod(pre_shape)
        samples = samples.reshape((n_element, sample_size))  # type: ignore
        samples_t = samples.T
        samples_t = (samples_t * samples_t[:, np.newaxis]).T
        cov_impact = -0.5 * (samples_t * cov).sum((1, 2))
        return np.concatenate([samples, cov_impact.reshape((n_element, 1))], 1).reshape(
            pre_shape + (sample_size + 1,)  # type: ignore
        )

    def param_to_T(param: ProbaParam) -> np.ndarray:
        # parametrised by the mean

        sigma = param[-1:]
        return np.concatenate([sigma ** (-2) * inv_cov @ param[:-1], sigma ** (-2)])

    def T_to_param(t_val: np.ndarray) -> ProbaParam:
        if t_val[-1] < 0.0:
            raise RenormError("Negative squared variance")

        return np.concatenate(
            [(t_val[-1] ** (-1)) * cov @ t_val[:-1], t_val[-1:] ** (-0.5)]
        )

    def g(t_val: np.ndarray) -> float:
        inv_sigma2 = t_val[-1]
        mu = t_val[:-1]
        return 0.5 * (
            -sample_size * np.log(inv_sigma2) + np.sum(mu * (cov @ mu)) / inv_sigma2
        )

    def der_g(t_val: np.ndarray) -> np.ndarray:
        mu, inv_sigma2 = t_val[:-1], t_val[-1]
        out = np.zeros(t_val.shape)
        out[:-1] = cov @ mu / inv_sigma2
        out[-1] = -0.5 * (
            sample_size / inv_sigma2 + np.sum(mu * (cov @ mu)) / (inv_sigma2**2)
        )
        return out

    return T, param_to_T, T_to_param, g, der_g, (t_dim,)


class FactCovGaussianMap(PreExpFamily):
    r"""
    Parametrization of gaussian distributions of form
            $(\mu, \sigma) -> N(\mu, \sigma^2 * Cov)$
    where Cov is fixed.

    The parameter is a 1D array of size d+1. The first d elements contain $\mu$, the last element
    specifies $\sigma$
    """

    # Indicate that this map deals with Gaussian
    map_type = "Gaussian"

    def __init__(
        self,
        sample_size: Optional[int] = None,
        sample_shape: Optional[tuple[int, ...]] = None,
        cov: Optional[ArrayLike] = None,
    ):
        sample_size, sample_shape = _shape_info(sample_size, sample_shape)

        if cov is None:
            cov = np.eye(sample_size)
        else:
            cov = np.array(cov)
            check_shape(cov, (sample_size, sample_size))

        T, param_to_T, T_to_param, g, der_g, (t_dim,) = exp_family_gauss_fact_cov(
            cov, sample_size, sample_shape
        )

        inv_cov = inverse_cov(cov)  # pre compute it once and for all

        def prob_map(x: ProbaParam) -> Gaussian:
            x = np.array(x)  # Force convert to array

            check_shape(x, (sample_size + 1,))  # type: ignore

            return Gaussian(
                means=x[:-1], cov=(x[-1] ** 2) * cov, sample_shape=sample_shape
            )

        def log_dens_der(x: ProbaParam) -> Callable[[Samples], np.ndarray]:
            """Signature: ProbaParam -> (Sample -> ProbaParam)"""
            x = np.array(x)
            means, sigma = x[:-1], x[-1]

            def derivative(samples: Samples) -> np.ndarray:
                pre_shape = _get_pre_shape(samples, sample_shape)  # type: ignore

                centered = samples.reshape(pre_shape + (sample_size,)) - means  # type: ignore
                der_means = (centered @ inv_cov) * sigma ** (-2)

                der_sigma = sigma ** (-1) * (
                    (centered * der_means).sum(-1) - sample_size
                )

                res = np.zeros((prod(pre_shape), sample_size + 1))  # type: ignore
                res[:, :-1] = der_means.flatten()
                res[:, -1] = der_sigma.flatten()

                return res.reshape(pre_shape + (sample_size + 1,))  # type: ignore

            return derivative

        ref_param = np.zeros(sample_size + 1)
        ref_param[-1] = 1

        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            T=T,
            param_to_T=param_to_T,
            T_to_param=T_to_param,
            g=g,
            der_g=der_g,
            ref_param=ref_param,
            proba_param_shape=(sample_size + 1,),
            sample_shape=sample_shape,
            t_shape=(t_dim,),
        )
        self._cov = cov
        self._inv_cov = inv_cov

    @property
    def cov(self):
        return self._cov

    @property
    def inv_cov(self):
        return self._inv_cov

    def __repr__(self) -> str:
        return str.join(
            "\n",
            [
                f"Gaussian Prior Map with covariance fixed up to a factor a on arrays of shape {self.sample_shape}.",
                f"Default covariance:\n{self.cov}",
            ],
        )

    def kl(self, param_1: ProbaParam, param_0: ProbaParam, n_sample: int = 0) -> float:
        """Computes the Kullback Leibler divergence between two gaussian distributions.
        defined by their prior parameters.

        Args:
            proba_1, proba_0 are 2 prior parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            kl(proba_1, proba_0)

        Computed as
            .5 * (2k * log(sigma0/sigma1)
            - k + k (sigma1/sigma0) **2
            + sigma0 ** (-2) diff_means * inv_cov @ diff_means
        """

        diff_means = param_0[:-1] - param_1[:-1]
        sigma0, sigma1 = param_0[-1], param_1[-1]
        ratio_sig = (sigma1 / sigma0) ** 2
        kl = (
            sigma0 ** (-2) * np.dot(diff_means, self.inv_cov @ diff_means)
            + (-np.log(ratio_sig) - 1 + ratio_sig) * self.sample_size
        ) / 2

        return kl

    def grad_kl(
        self, param_0: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        (nabla_{param_1} kl(param_1, param_0))

        Args:
            param_1, param_0 are 2 distribution parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            nabla_{param_1}kl(param_1, param_0)
            See doc for more information.
        """
        inv_cov = self.inv_cov
        means0 = param_0[:-1]
        sigma0 = param_0[-1]

        def fun(
            param_1: ProbaParam,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> tuple[ProbaParam, float]:
            diff_mean = param_1[:-1] - means0
            sigma1 = param_1[-1]

            grad_kl_mean = sigma0 ** (-2) * inv_cov @ diff_mean

            ratio_sig = (sigma1 / sigma0) ** 2
            grad_ratio_sig = 2 * ratio_sig / sigma1

            grad_kl_sig = grad_ratio_sig * (-1 / ratio_sig + 1) * self._sample_size

            der = np.zeros(self._sample_size + 1)
            der[:-1] = grad_kl_mean
            der[-1] = 0.5 * grad_kl_sig

            return der, 0.5 * (
                np.dot(diff_mean, grad_kl_mean)
                + (-np.log(ratio_sig) - 1 + ratio_sig) * self._sample_size
            )

        return fun

    def grad_right_kl(
        self, param_1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        (nabla_{param_1} kl(param_1, param_0))

        Args:
            param_1, param_0 are 2 distribution parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            nabla_{param_1}kl(param_1, param_0)
            See doc for more information.
        """
        inv_cov = self.inv_cov
        means1 = param_1[:-1]
        sigma1 = param_1[-1]

        def fun(
            param_0: ProbaParam,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> tuple[ProbaParam, float]:
            diff_mean = param_0[:-1] - means1
            sigma0 = param_0[-1]

            grad_kl_mean = sigma0 ** (-2) * inv_cov @ diff_mean

            ratio_sig = (sigma1 / sigma0) ** 2
            grad_ratio_sig = -2 * ratio_sig / sigma0

            grad_kl_sig = grad_ratio_sig * (-1 / ratio_sig + 1) * self._sample_size

            der = np.zeros(self._sample_size + 1)
            der[:-1] = grad_kl_mean
            der[-1] = 0.5 * grad_kl_sig - np.dot(diff_mean, grad_kl_mean) / sigma0

            return der, 0.5 * (
                np.dot(diff_mean, grad_kl_mean)
                + (-np.log(ratio_sig) - 1 + ratio_sig) * self._sample_size
            )

        return fun
