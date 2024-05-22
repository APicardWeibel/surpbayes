""" Helper functions for change of parametrisation + KL computations 
of block diagonal covariance gaussians

__FUTURE__
functions in this module should be compiled at package installation.
This was previously done through numba.njit decorator, but deactivated
since this slowed up massively package loading (need to compile every
time!).

The current perspective is to wait until numba's AOT capabilities have
stabilized (AOT is pending deprecation in current version 0.58.1), then
use the new framework to perform AOT.

In the meanwhile, users ready to suffer from circ. 30s overhead at
package loading can uncomment the numba signatures and numba imports
"""

# import numba as nb
import numpy as np
# import surpbayes.nb_typ as nbt
from surpbayes.proba._errors import RenormError
from surpbayes.proba.gauss.Gauss.nb_helper import _kl, make_cov


# @nb.njit(nbt.Tuple((nbt.f1D, nbt.f2D))(nbt.f1D, nbt.i, nbt.i1D))
def _T_to_quadra_loc(t_val_loc: np.ndarray, sample_size: int, good_index: np.ndarray):
    H = np.zeros(sample_size**2)
    # Set non diagonal values
    H[good_index] = t_val_loc[(2 * sample_size) :]
    H = H.reshape((sample_size, sample_size))
    H = H + H.T + np.diag(t_val_loc[sample_size : (2 * sample_size)])
    # Compute center
    center = np.linalg.inv(H) @ t_val_loc[0:sample_size]

    return (center, H)


# @nb.njit(nbt.f1D(nbt.f1D_C, nbt.i1D, nbt.i, nbt.i))
def loc_param_to_T(
    loc_param: np.ndarray, good_index: list[int], sample_size: int, t_dim: int
) -> np.ndarray:

    mean = loc_param[:sample_size]
    half_cov = loc_param[sample_size:].reshape((sample_size, sample_size))
    inv_cov = np.linalg.inv(make_cov(half_cov))
    trans_mean = inv_cov @ mean

    output = np.zeros(t_dim)
    output[:sample_size] = trans_mean
    output[sample_size : (2 * sample_size)] = np.diag(inv_cov)
    output[(2 * sample_size) :] = inv_cov.flatten()[good_index]

    return output


# @nb.njit(nbt.f1D(nbt.f1D, nbt.f2D, nbt.i))
def _quadra_to_param(center, H, sample_size) -> np.ndarray:
    vals, vects = np.linalg.eigh(H)
    if vals[0] < 0.0:
        raise RenormError("Inverse covariance must be positive")

    half_cov = np.sqrt(1 / vals) * vects
    accu = np.zeros((sample_size + 1, sample_size))
    accu[0] = center
    accu[1:] = half_cov
    return accu.flatten()


# @nb.njit(nbt.Tuple((nbt.f1D, nbt.f2D))(nbt.f1D, nbt.i1D, nbt.i))
def _par_to_mu_M(
    loc_par: np.ndarray, good_index: np.ndarray, sample_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mu = Cov^{-1} mean and M = Cov^{-1} from parameter"""
    mu = loc_par[:sample_size]

    M_flat = np.zeros(sample_size**2)
    M_flat[good_index] = loc_par[2 * sample_size :]
    M = M_flat.reshape((sample_size, sample_size))
    M = M + M.T + np.diag(loc_par[sample_size : (2 * sample_size)])
    return mu, M


# @nb.njit(nbt.Tuple((nbt.f1D, nbt.f2D))(nbt.f1D, nbt.f2D))
def __der_g(mu: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv_M = np.linalg.inv(M)
    d_g_mu = inv_M @ mu
    d_g_M = -0.5 * inv_M - 0.5 * np.outer(d_g_mu, d_g_mu)
    return d_g_mu, d_g_M


# @nb.njit(nbt.f1D(nbt.f1D, nbt.i1D, nbt.i))
def _pre_der_g(loc_par: np.ndarray, good_index: np.ndarray, sample_size: int):
    d_g_mu, d_g_M = __der_g(*_par_to_mu_M(loc_par, good_index, sample_size))
    d_par = np.zeros(loc_par.shape)
    d_par[:sample_size] = d_g_mu
    d_par[sample_size : (2 * sample_size)] = np.diag(d_g_M)
    d_par[(2 * sample_size) :] = 2 * d_g_M.flatten()[good_index]
    return d_par


# @nb.njit(nbt.f1D(nbt.f1D, nbt.f1D, nbt.i1D, nbt.i1D))
def _block_kl(param_1: np.ndarray, param_0: np.ndarray, ds: np.ndarray, cuts: np.ndarray):
    """
    Computes the Kullback Leibler divergence between two Block diagonal
    gaussian distributions defined by their meta parameters.

    This function is numba.njit decorated.

    Args:
        proba_1, proba_0 are 2 meta parameters
        dim: number of dimensions

    Output:
        kl(proba_1, proba_0)
    """
    s = np.zeros(len(ds))
    param_0_c = np.ascontiguousarray(param_0)
    param_1_c = np.ascontiguousarray(param_1)

    for i, d in enumerate(ds):
        a = cuts[i]
        b = cuts[i + 1]

        par0_loc = param_0_c[a:b].reshape((d + 1, d))
        par1_loc = param_1_c[a:b].reshape((d + 1, d))

        s[i] = _kl(par1_loc, par0_loc, d)
    return s
