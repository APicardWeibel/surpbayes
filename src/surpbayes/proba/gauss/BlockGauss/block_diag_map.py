# import warnings
from math import log, pi
from typing import Callable

import numpy as np
from surpbayes.misc import _get_pre_shape, prod
from surpbayes.proba.exponential_family.pre_exponential_family import PreExpFamily
from surpbayes.proba.gauss.BlockGauss.block_diag import BlockDiagGauss
from surpbayes.proba.gauss.BlockGauss.helper import check_blocks
from surpbayes.proba.gauss.BlockGauss.nb_helper import (
    _block_kl,
    _par_to_mu_M,
    _pre_der_g,
    _quadra_to_param,
    _T_to_quadra_loc,
    loc_param_to_T,
)
from surpbayes.proba.gauss.Gauss import GaussianMap, inverse_cov, make_cov
from surpbayes.proba.gauss.Gauss.gaussian_map import _comp_log_dens_der
from surpbayes.types import ProbaParam, Samples


def split_prob_param_blocks(
    blocks: list[list[int]],
) -> Callable[[ProbaParam], list[ProbaParam]]:
    ds = [len(block) for block in blocks]
    param_lens = [d**2 + d for d in ds]
    proba_param_cuts = np.cumsum([0] + param_lens)

    def split(param: ProbaParam) -> list[ProbaParam]:
        return [
            param[a:b].reshape((d + 1, d))
            for d, a, b in zip(ds, proba_param_cuts, proba_param_cuts[1:])
        ]

    return split


def exp_family_block_gauss(blocks: list[list[int]]):
    r"""Prepare functions to interpret block diagonal gaussian distributions as exponential family

    Compute the T for exponential family interpretation of Gaussian distributions.

    Should be coherent with versions implemented for Gaussian distributions.
    """

    dim_list = [len(block) for block in blocks]
    tot_dim = sum(dim_list)

    t_lengths = [int(d * (d + 3) / 2) for d in dim_list]
    t_stops = np.cumsum([0] + t_lengths)
    tot_t = sum(t_lengths)

    def get_good_index(sample_size) -> list[int]:
        good_index = []
        for i in range(sample_size):
            good_index += [i * sample_size + a for a in range(i + 1, sample_size)]
        return np.array(good_index, dtype=int)

    good_indexes_list = [get_good_index(sample_size) for sample_size in dim_list]

    def T(xs: np.ndarray):
        pre_shape = xs.shape[:-1]
        n_elements = prod(pre_shape)  # type: ignore

        xs = xs.reshape((n_elements, tot_dim))

        def loc_T(xs_red: np.ndarray, good_indexes: list[int]) -> np.ndarray:
            xs_red_t = xs_red.T
            sample_size = len(xs_red_t)
            xs_tensor = (xs_red_t[:, np.newaxis] * xs_red_t).reshape(
                (sample_size**2, n_elements)
            )
            xs_tensor = xs_tensor[good_indexes].transpose()

            return np.concatenate([xs_red, -0.5 * xs_red**2, -xs_tensor], -1)

        return np.concatenate(
            [
                loc_T(xs_red=xs[:, block], good_indexes=gi)
                for block, gi in zip(blocks, good_indexes_list)
            ],
            -1,
        ).reshape(pre_shape + (tot_t,))

    par_dim = [d * (d + 1) for d in dim_list]
    par_stops = np.cumsum([0] + par_dim)

    def param_to_T(param: ProbaParam) -> np.ndarray:
        param_c = np.ascontiguousarray(param)
        return np.concatenate(
            [
                loc_param_to_T(
                    loc_param=param_c[a:b],
                    good_index=good_index,
                    sample_size=d,
                    t_dim=t_dim,
                )
                for a, b, good_index, d, t_dim in zip(
                    par_stops, par_stops[1:], good_indexes_list, dim_list, t_lengths
                )
            ]
        )

    t_lengths = [(d * (d + 3)) // 2 for d in dim_list]
    t_stops = np.cumsum([0] + t_lengths)

    def T_to_param(t_val: np.ndarray) -> ProbaParam:

        quadras = [
            _T_to_quadra_loc(t_val[a:b], sample_size, good_index)
            for a, b, sample_size, good_index in zip(
                t_stops, t_stops[1:], dim_list, good_indexes_list
            )
        ]

        return np.concatenate(
            [
                _quadra_to_param(quadra[0], quadra[1], d)
                for quadra, d in zip(quadras, dim_list)
            ]
        )

    def par_to_mu_M_s(par):

        return [
            _par_to_mu_M(par[b:e], gidx, sdim)
            for b, e, gidx, sdim in zip(
                t_stops, t_stops[1:], good_indexes_list, dim_list
            )
        ]

    def log_det_M(M):
        return log(np.linalg.det(M))

    def _g(mu_M_s):
        return np.sum(
            [
                0.5
                * (
                    sdim * log(2 * pi)
                    - log_det_M(M)
                    + np.sum(mu * (np.linalg.inv(M) @ mu))
                )
                for (mu, M), sdim in zip(mu_M_s, dim_list)
            ]
        )

    def g(par):
        return _g(par_to_mu_M_s(par))

    def der_g(par):
        return np.concatenate(
            [
                _pre_der_g(par[b:e], gidx, sdim)
                for b, e, gidx, sdim in zip(
                    t_stops, t_stops[1:], good_indexes_list, dim_list
                )
            ]
        )

    return T, param_to_T, T_to_param, g, der_g, (tot_t,)


class BlockDiagGaussMap(PreExpFamily):
    """
    Block diagonal gaussian family.
    """

    # Indicate that this map deals with Gaussian
    map_type = "Gaussian"

    def __init__(self, blocks: list[list[int]]):

        check_blocks(blocks)

        self.ds = np.array([len(block) for block in blocks])
        param_lens = [d**2 + d for d in self.ds]
        self.proba_param_cuts = np.cumsum([0] + param_lens)

        splitter = split_prob_param_blocks(blocks)

        def _get_mean_cov(
            param: ProbaParam,
        ) -> tuple[list[ProbaParam], list[np.ndarray]]:
            split_params = splitter(param)
            means = [par[0] for par in split_params]
            pre_covs = [par[1:] for par in split_params]
            covs = [make_cov(pre_cov) for pre_cov in pre_covs]
            return means, covs

        dim_list = [len(block) for block in blocks]
        sample_size = sum(dim_list)

        sample_shape = (sample_size,)
        proba_param_shape = (sum(len(block) * (len(block) + 1) for block in blocks),)

        def prob_map(param: ProbaParam) -> BlockDiagGauss:
            split_params = splitter(param)
            means = [par[0] for par in split_params]
            covs = [make_cov(par[1:]) for par in split_params]
            return BlockDiagGauss(means, covs, blocks, check=False)

        def log_dens_der(param: ProbaParam) -> Callable[[Samples], np.ndarray]:
            split_params = splitter(param)
            means = [par[0] for par in split_params]
            pre_covs = [par[1:] for par in split_params]
            covs = [make_cov(pre_cov) for pre_cov in pre_covs]
            inv_covs = [inverse_cov(cov) for cov in covs]

            pre_comp_grads = [
                -inv_cov @ pre_cov for inv_cov, pre_cov in zip(inv_covs, pre_covs)
            ]

            def derivative(samples: Samples) -> np.ndarray:
                pre_shape = _get_pre_shape(samples, sample_shape)
                pre_dim = prod(pre_shape)
                samples = samples.reshape((pre_dim,) + sample_shape)

                return np.concatenate(
                    [
                        _comp_log_dens_der(
                            samples[:, block],
                            len(block),
                            loc_mean,
                            inv_cov,
                            pre_cov,
                            pre_comp_grad,
                        ).reshape((pre_dim, len(block) * (len(block) + 1)))
                        for block, loc_mean, inv_cov, pre_comp_grad, pre_cov in zip(
                            blocks, means, inv_covs, pre_comp_grads, pre_covs
                        )
                    ],
                    axis=1,
                ).reshape(pre_shape + proba_param_shape)

            return derivative

        ref_param = np.concatenate(
            [
                np.concatenate([np.zeros(len(block)), np.eye(len(block)).flatten()])
                for block in blocks
            ]
        )

        (T, param_to_T, T_to_param, g, der_g, t_shape) = exp_family_block_gauss(blocks)

        super().__init__(
            prob_map,
            log_dens_der,
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
        self.blocks = blocks
        self.splitter = splitter
        self._get_mean_cov = _get_mean_cov
        self._dim_list = dim_list
        self._gmaps = [
            GaussianMap(d) for d in self._dim_list
        ]  # For KL/Grad_kl/grad_right_kl computations

    def __repr__(self) -> str:
        return str.join(
            "\n",
            [
                f"Map for Block Diagonal Gaussian distributions with blocks {self.blocks}."
            ],
        )

    def kl_blocks(self, param_1: ProbaParam, param_0: ProbaParam) -> np.ndarray:
        return _block_kl(param_1, param_0, self.ds, self.proba_param_cuts)

    def kl(self, param_1: ProbaParam, param_0: ProbaParam, n_sample: int = 0) -> float:
        """Computes the Kullback Leibler divergence between two gaussian distributions.
        defined by their meta parameters.

        Args:
            proba_1, proba_0 are 2 meta parameters
            n_sample is disregarded (exact computations used instead)

        Output:
            kl(proba_1, proba_0)
        """
        if np.array_equal(param_1, param_0):
            return 0.0
        return np.sum(self.kl_blocks(param_1, param_0))

    def grad_kl(
        self, param_0: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        param_0_s = self.splitter(param_0)
        pre_grad_kls = [
            gmap.grad_kl(par0) for gmap, par0 in zip(self._gmaps, param_0_s)  # type: ignore
        ]

        def fun(param_1: np.ndarray, n_sample: int = 0):  # pylint: disable=W0613
            param_1_s = self.splitter(param_1)
            pre_grads = [
                pre_grad_kl(par1) for pre_grad_kl, par1 in zip(pre_grad_kls, param_1_s)  # type: ignore
            ]
            kl = sum([pre_grad[1] for pre_grad in pre_grads])
            grad = np.concatenate([pre_grad[0].flatten() for pre_grad in pre_grads])
            return grad, kl

        return fun

    def grad_right_kl(
        self, param_1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        param_1_s = self.splitter(param_1)
        pre_grad_kls = [
            gmap.grad_right_kl(par1) for gmap, par1 in zip(self._gmaps, param_1_s)
        ]

        def fun(param_0: np.ndarray, n_sample: int = 0):  # pylint: disable=W0613
            param_0_s = self.splitter(param_0)
            pre_grads = [
                pre_grad_kl(par0) for pre_grad_kl, par0 in zip(pre_grad_kls, param_0_s)  # type: ignore
            ]
            kl = sum([pre_grad[1] for pre_grad in pre_grads])
            grad = np.concatenate([pre_grad[0].flatten() for pre_grad in pre_grads])
            return grad, kl

        return fun
