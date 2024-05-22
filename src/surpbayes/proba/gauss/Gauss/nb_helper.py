""" Helper functions for change of parametrisation for Gaussian
distributions.

__FUTURE__
functions in this module should be compiled at package installation.
This was previously done through numba.njit decorator, but deactivated
since this slowed up massively package loading (need to compile every
time!).

The current perspective is to wait until numba's AOT capabilities have
stabilized (AOT is pending deprecation in current version 0.58.1), then
use the new framework to perform AOT.

In the meanwhile, users ready to suffer from circ. 30s overhead at
package loading can uncomment the numba signatures and numba imports +
use the catch_warnings() environment to avoid "NumbaPerformanceWarning"
"""

# import warnings
from math import log, pi

# import numba as nb
import numpy as np
# import surpbayes.nb_typ as nbt
from surpbayes.proba._errors import RenormError

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=nb.NumbaPerformanceWarning)

#     @nb.njit(nbt.f2D(nbt.f1D, nbt.i, nbt.i1D))
def _T_to_param(
    t_val: np.ndarray, sample_size: int, good_indexes: np.ndarray
) -> np.ndarray:
    H = np.zeros(sample_size**2)  # type: ignore
    # Set non diagonal values
    H[good_indexes] = t_val[(2 * sample_size) :]  # type: ignore
    H = H.reshape((sample_size, sample_size))  # type: ignore
    H = H + H.T + np.diag(t_val[sample_size : (2 * sample_size)])  # type: ignore

    # Compute center
    center = (
        np.linalg.inv(H) @ t_val[:sample_size]
    ) 

    vals, vects = np.linalg.eigh(H)
    if vals[0] < 0:
        raise RenormError("Inverse covariance must be positive")

    half_cov = np.sqrt(1 / vals) * vects
    accu = np.zeros((sample_size + 1, sample_size))  # type: ignore
    accu[0] = center
    accu[1:] = half_cov
    return accu


# @nb.njit(nbt.Tuple((nbt.f1D, nbt.f2D))(nbt.f1D, nbt.i, nbt.i1D))
def par_to_mu_M(
    t_par: np.ndarray, sample_size: int, good_indexes: np.ndarray
) -> np.ndarray:

    M_flat = np.zeros(sample_size**2)
    M_flat[good_indexes] = t_par[2 * sample_size :]
    M = M_flat.reshape((sample_size, sample_size))
    M = M + M.T + np.diag(t_par[sample_size : (2 * sample_size)])
    return t_par[:sample_size], M


# @nb.njit(nbt.f(nbt.f2D))
def log_det_M(M):
    return log(np.linalg.det(M))


# @nb.njit(nbt.f(nbt.f1D, nbt.f2D, nbt.i))
def _g_mu_M(mu, M, sample_size):
    return 0.5 * (
        sample_size * log(2 * pi) - log_det_M(M) + np.sum(mu * (np.linalg.inv(M) @ mu))
    )


# @nb.njit(nbt.f(nbt.f1D, nbt.i, nbt.i1D))
def _g(t_par: np.ndarray, sample_size: int, good_indexes: np.ndarray):
    mu, M = par_to_mu_M(t_par, sample_size, good_indexes)
    return _g_mu_M(mu=mu, M=M, sample_size=sample_size)


# @nb.njit(nbt.Tuple((nbt.f1D, nbt.f2D))(nbt.f1D, nbt.f2D))
def _der_g_mu_M(mu: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv_M = np.linalg.inv(M)
    d_g_mu = inv_M @ mu
    d_g_M = -0.5 * inv_M - 0.5 * np.outer(d_g_mu, d_g_mu)
    return d_g_mu, d_g_M


# @nb.njit(nbt.f1D(nbt.f1D, nbt.i, nbt.i1D))
def _der_g(
    t_par: np.ndarray, sample_size: int, good_indexes: np.ndarray
) -> np.ndarray:
    mu, M = par_to_mu_M(t_par, sample_size=sample_size, good_indexes=good_indexes)
    d_g_mu, d_g_M = _der_g_mu_M(mu, M)
    d_par = np.zeros(t_par.shape)
    d_par[:sample_size] = d_g_mu
    d_par[sample_size : (2 * sample_size)] = np.diag(d_g_M)
    d_par[(2 * sample_size) :] = 2 * d_g_M.flatten()[good_indexes]
    return d_par


# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=nb.NumbaPerformanceWarning)

    # @nb.njit(nbt.f2D(nbt.f2Dru))
def make_cov(pre_cov: np.ndarray) -> np.ndarray:
    """Convert encoding of covariance to covariance. Used for GaussianMap"""
    return pre_cov @ pre_cov.T

# @nb.njit(nbt.f(nbt.f2D, nbt.f2D, nbt.i))
def _kl(par1: np.ndarray, par0: np.ndarray, dim: int):
    """
    Computes the Kullback Leibler divergence between two gaussian distributions.
    defined by their meta parameters.

    This function is numba.njit decorated.

    Args:
        proba_1, proba_0 are 2 meta parameters
        dim: number of dimensions

    Output:
        kl(proba_1, proba_0)

    """
    delta = par0[0] - par1[0]

    cov0 = make_cov(par0[1:])
    cov1 = make_cov(par1[1:])

    inv_cov0 = np.linalg.inv(cov0)
    a = inv_cov0 @ cov1

    kl = -np.log(np.linalg.det(a))
    kl = kl - dim + np.sum(np.diag(a))

    kl = kl + np.sum(delta * (inv_cov0 @ delta))
    return kl / 2

# @nb.njit(nbt.Tuple((nbt.f2D, nbt.f))(nbt.f2D, nbt.f1D, nbt.f2D, nbt.i, nbt.f2D))
def _grad_kl(
    param_0: np.ndarray,
    means_0: np.ndarray,
    inv_cov_0: np.ndarray,
    dim: int,
    param_1: np.ndarray,
):
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

    if np.array_equal(param_0, param_1):
        return np.zeros((dim + 1, dim)), 0.0

    der = np.zeros((dim + 1, dim))

    means_1 = param_1[0]
    pre_cov = param_1[1:]

    cov_1 = make_cov(pre_cov)
    inv_cov_1 = np.linalg.inv(cov_1)

    delta = means_1 - means_0

    der[0] = inv_cov_0 @ (delta)  # type: ignore
    grad_cov = 0.5 * (inv_cov_0 - inv_cov_1)  # type: ignore

    grad_param = 2 * grad_cov @ pre_cov
    der[1:] = grad_param

    a = inv_cov_0 @ cov_1
    kl = -np.log(np.linalg.det(a))
    kl = kl - dim + np.sum(np.diag(a))
    kl = kl + np.sum(delta * (inv_cov_0 @ delta))

    return der, kl / 2

# @nb.njit(nbt.Tuple((nbt.f2D, nbt.f))(nbt.f2D, nbt.f1D, nbt.f2D, nbt.i, nbt.f2D))
def _grad_right_kl(
    param_1: np.ndarray,
    means_1: np.ndarray,
    cov_1: np.ndarray,
    dim: int,
    param_0: np.ndarray,
):

    if np.array_equal(param_0, param_1):
        return np.zeros((dim + 1, dim)), 0.0

    means_0 = param_0[0]
    pre_cov = param_0[1:]
    cov_0 = make_cov(pre_cov)
    inv_cov_0 = np.linalg.inv(cov_0)

    der = np.zeros((dim + 1, dim))

    delta = means_0 - means_1
    der[0] = inv_cov_0 @ delta  # type: ignore

    a = inv_cov_0 @ cov_1

    grad_cov = 0.5 * (
        inv_cov_0  # type: ignore
        - a @ inv_cov_0  # type: ignore
        - np.outer(der[0], der[0])
    )

    grad_param = 2 * grad_cov @ pre_cov
    der[1:] = grad_param

    kl = -np.log(np.linalg.det(a))
    kl = kl - dim + np.sum(np.diag(a))
    kl = kl + np.sum(delta * (inv_cov_0 @ delta))

    return der, kl / 2
