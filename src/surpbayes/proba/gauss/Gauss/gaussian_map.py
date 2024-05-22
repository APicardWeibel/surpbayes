"""
GaussianMap submodule
"""

import warnings
from typing import Callable, Optional

import numpy as np
from surpbayes.misc import _get_pre_shape, prod
from surpbayes.proba._helper import _shape_info
from surpbayes.proba.exponential_family.pre_exponential_family import PreExpFamily
from surpbayes.proba.gauss.Gauss.gaussian import Gaussian
from surpbayes.proba.gauss.Gauss.nb_helper import (_der_g, _g, _grad_kl,
                                              _grad_right_kl, _kl, _T_to_param,
                                              make_cov)
from surpbayes.proba.warnings import ShapeWarning
from surpbayes.types import ProbaParam, Samples


def inverse_cov(
    cov: np.ndarray,
) -> np.ndarray:
    """Safe inversion of covariance.
    TO DO: modify and put in misc (can be useful in general)"""
    vals, vects = np.linalg.eigh(cov)
    inv_vals = np.array([val**-1 if val > 0.0 else 0.0 for val in vals])

    inv_cov = (inv_vals * vects) @ vects.T
    return inv_cov


def _comp_log_dens_der(
    samples: np.ndarray,
    sample_size: int,
    means: np.ndarray,
    inv_cov: np.ndarray,
    pre_cov: np.ndarray,
    pre_compute_grad,
) -> np.ndarray:
    """
    Takes a list of samples as input, outputs a list of gradients (array like list of proba param)
    """
    n_samples = len(samples)

    centered = (
        samples.reshape(
            (
                n_samples,
                sample_size,
            )
        )
        - means
    )  # Shape (pre_dim, sample_size)

    der_0 = centered @ inv_cov  # Shape (pre_dim, sample_size)

    der_0_t = der_0.T
    outer_prod = (der_0_t[:, np.newaxis] * der_0_t).T
    grad_param = pre_compute_grad + outer_prod @ pre_cov
    der_1 = grad_param

    return np.concatenate([der_0[:, np.newaxis], der_1], 1)


def exp_family_gauss(
    sample_size: Optional[int] = None,
    sample_shape: Optional[tuple[int, ...]] = None,
) -> tuple[Callable, Callable, Callable, Callable, Callable, tuple[int, ...]]:
    r"""Prepare functions to interpret gaussian distributions as exponential family

    Compute the T for exponential family interpretation of Gaussian distributions.

    IMPORTANT: The parametrisation $\theta$ of the exponential family using the T is NOT
    the parametrisation used in the GaussianMap class. Note that the parametrisation used
    in the GaussianMap class can not be used for any exponential family interpretation of
    gaussians, as the covariance is mapped in a non linear fashion (incentive: the parametrisation
    works on $R^{(d+1, d)}$).

    The output T works for the following parametrisation of a gaussian distribution:
        $(Cov @ mean, (Cov_{i,i})_i, (Cov_{i,j})_{i>j})$.

    As such, x is mapped to
        $(x, -.5 x^2, -(x_i x_j)_{i>j})$

    Then $A . T(x)  = - .5 \tilde{A} .x x^t + A_0^t x
                    = - .5 x^t \tilde{A} x + A_0^t \tilde{A}^{-1} \tilde{A} x
                    = -.5 (x - \tilde{A}^{-1} A_0) \tilde{A} (x - \tilde{A}^{-1} A_0)$
    """

    sample_size, sample_shape = _shape_info(sample_size, sample_shape)  # type: ignore
    good_indexes = []
    for i in range(sample_size):
        good_indexes += [i * sample_size + a for a in range(i + 1, sample_size)]
    good_indexes = np.array(good_indexes, dtype=int)
    t_dim = 2 * sample_size + len(good_indexes)

    def T(samples: Samples) -> np.ndarray:
        pre_shape = _get_pre_shape(samples, sample_shape)  # type: ignore
        n_elements = prod(pre_shape)
        samples = samples.reshape((n_elements, sample_size))  # type: ignore

        samples_t = samples.T  # shape (d, n)
        samples_tensor = (samples_t[:, np.newaxis] * samples_t).reshape(
            (sample_size**2, n_elements)  # type: ignore
        )
        samples_tensor = samples_tensor[good_indexes].transpose()

        return np.concatenate(
            [samples, -0.5 * samples**2, -samples_tensor], -1
        ).reshape(pre_shape + (t_dim,))

    def param_to_T(param: ProbaParam) -> np.ndarray:
        mean = param[0]
        half_cov = param[1:]
        inv_cov = np.linalg.inv(make_cov(half_cov))
        trans_mean = inv_cov @ mean
        return np.concatenate(
            [trans_mean, np.diag(inv_cov), inv_cov.flatten()[good_indexes]], -1
        )

    def T_to_param(t_par: np.ndarray) -> ProbaParam:
        return _T_to_param(t_par, sample_size, good_indexes)

    def g(t_par):
        return _g(t_par, sample_size, good_indexes)

    def der_g(t_par: np.ndarray) -> np.ndarray:
        return _der_g(t_par, sample_size, good_indexes)

    return T, param_to_T, T_to_param, g, der_g, (t_dim,)


class GaussianMap(PreExpFamily):
    """
    For Gaussian distributions, use the following subclass (KL/grad_KL is overriden)

    The distribution param shape is (pred_param_len + 1, pred_param_len)
        - param[0] gives the mean,
        - param[1:] is a matrix m defining the covariance through $cov = m * m^T$

    The covariance is not used as a parameter to simplify routines such as gradient descents.
    """

    # Indicate that this map deals with Gaussian
    map_type = "Gaussian"

    def __init__(
        self,
        sample_size: Optional[int] = None,
        sample_shape: Optional[tuple[int, ...]] = None,
    ):
        """
        Define a GaussianMap on vectors of known shape.

        Either sample_size or sample_shape must be specified. If both are, sample_size is ignored.
        """

        sample_size, sample_shape = _shape_info(sample_size, sample_shape)  # type: ignore

        def prob_map(x: ProbaParam) -> Gaussian:
            x = np.array(x)  # Force convert to array
            if x.shape != (sample_size + 1, sample_size):  # type: ignore
                warnings.warn(
                    "\n".join(
                        [
                            f"Distribution parameter shape is {x.shape} (Expected{(sample_size+1,sample_size)})",  # type: ignore
                            "Trying to construct nonetheless. Watch out for strange behaviour.",
                        ]
                    ),
                    category=ShapeWarning,
                )
            means = x[0]
            pre_cov = x[1:]
            cov = make_cov(pre_cov)
            return Gaussian(means=means, cov=cov, sample_shape=sample_shape)

        proba_param_shape = (sample_size + 1, sample_size)

        def log_dens_der(
            x: ProbaParam,
        ) -> Callable[[np.ndarray], np.ndarray]:
            means = x[0]
            pre_cov = x[1:]
            cov = make_cov(pre_cov)

            inv_cov = inverse_cov(cov)

            pre_compute_grad = -inv_cov @ pre_cov

            def derivative(samples: np.ndarray) -> np.ndarray:
                # ys is a np.ndarray with shape ending in sample shape
                pre_shape = _get_pre_shape(samples, sample_shape)  # type: ignore
                pre_dim = prod(pre_shape)
                return _comp_log_dens_der(
                    samples.reshape((pre_dim, sample_size)),  # type: ignore
                    sample_size,
                    means,
                    inv_cov,
                    pre_cov,
                    pre_compute_grad,
                ).reshape(pre_shape + proba_param_shape)

            return derivative

        # Construct reference parameter (standard normal)
        ref_param = np.zeros((sample_size + 1, sample_size))
        ref_param[1:] = np.eye(sample_size)
        (T, param_to_T, T_to_param, g, der_g, t_shape) = exp_family_gauss(
            sample_size, sample_shape
        )

        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            T=T,
            param_to_T=param_to_T,
            T_to_param=T_to_param,
            g=g,
            der_g=der_g,
            ref_param=ref_param,
            proba_param_shape=proba_param_shape,
            sample_shape=sample_shape,
            t_shape=t_shape,
        )

    def __repr__(self) -> str:
        return str.join(
            "\n", [f"Gaussian Prior Map on arrays of shape {self.sample_shape}."]
        )

    def kl(self, param_1: ProbaParam, param_0: ProbaParam, n_sample: int = 0) -> float:
        """Computes the Kullback Leibler divergence between two gaussian distributions.
        defined by their prior parameters.

        Args:
            proba_1, proba_0 are 2 prior parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            kl(proba_1, proba_0)
        """
        return _kl(param_1, param_0, self._sample_size)

    def grad_kl(
        self, param_0: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        (nabla_{param_1} KL(param_1, param_0))

        Args:
            param_1, param_0 are 2 distribution parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            nabla_{param_1}KL(param_1, param_0)
        """

        means_0 = param_0[0]
        cov_0 = make_cov(param_0[1:])
        inv_cov_0 = np.linalg.inv(cov_0)

        def fun(param_1: np.ndarray, n_sample: int = 0):  # pylint: disable=W0613
            return _grad_kl(
                param_0,
                means_0,
                inv_cov_0,
                self._sample_size,
                param_1,
            )

        return fun

    def grad_right_kl(
        self, param_1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:

        means_1 = param_1[0]
        cov_1 = make_cov(param_1[1:])

        def fun(
            param_0: ProbaParam,
            n_sample: int = 0,  # pylint: disable=W0613
        ) -> tuple[ProbaParam, float]:
            return _grad_right_kl(
                param_1,
                means_1,
                cov_1,
                self._sample_size,
                param_0,
            )

        return fun

    def to_param(self, g_proba: Gaussian) -> np.ndarray:
        """
        Transforms a Gaussian back to a np.ndarray such that
        self(self.to_param(g_proba)) = g_proba
        """
        means = g_proba.means
        vals, vects = g_proba.vals, g_proba.vects
        accu = np.zeros(self._proba_param_shape)
        accu[0] = means
        accu[1:] = np.sqrt(vals) * vects
        return accu
