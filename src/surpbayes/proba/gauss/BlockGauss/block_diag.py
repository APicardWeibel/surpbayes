from typing import Optional, Sequence

import numpy as np
from surpbayes.proba.gauss.BlockGauss.helper import check_blocks
from surpbayes.proba.gauss.Gauss import Gaussian


def check_coherence(
    means: Sequence[np.ndarray], covs: Sequence[np.ndarray], blocks: list[list[int]]
):
    for mu, cov, block in zip(means, covs, blocks):
        d = len(mu)
        assert cov.shape == (d, d)
        assert len(block) == d

def inv_permut(permut):
    """Fast invert a permutation
    (code from https://stackoverflow.com/questions/9185768/inverting-permutations-in-python)"""
    inv = np.empty_like(permut)
    inv[permut] = np.arange(len(inv), dtype=inv.dtype)
    return inv

class BlockDiagGauss(Gaussian):
    def __init__(
        self,
        means: Sequence[np.ndarray],
        covs: Sequence[np.ndarray],
        blocks: Optional[list[list[int]]] = None,
        check: bool = True,
    ):
        lens = [0] + [len(mu) for mu in means]
        lens = np.cumsum(lens)  # type: ignore

        if blocks is None:
            # Order not specified => no ordering
            reorder = False
            blocks = [list(range(a, b)) for a, b in zip(lens, lens[1:])]
        else:
            reorder = True
            order = inv_permut([i for block in blocks for i in block])
        if check:
            check_blocks(blocks)
            check_coherence(means, covs, blocks)

        d_tot = sum(len(mu) for mu in means)
        cov_tot = np.zeros((d_tot, d_tot))
        inv_covs = [np.linalg.inv(cov) for cov in covs]
        vects, vals = np.zeros((d_tot, d_tot)), np.zeros(d_tot)

        means_tot = np.concatenate(means)

        for a, b, cov in zip(lens, lens[1:], covs):
            cov_tot[a:b, a:b] = cov
            loc_vals, loc_vects = np.linalg.eigh(cov)
            vects[a:b, a:b] = loc_vects
            vals[a:b] = loc_vals

        sorter = np.argsort(vals)
        vals = vals[sorter]
        vects = vects[:, sorter]
        if reorder:
            cov_tot = cov_tot[order][:, order]
            vects = vects[order]
            means_tot = means_tot[order]

        super().__init__(
            means=means_tot,
            cov=cov_tot,
            info={"vals": vals, "vects": vects},
            sample_shape=(d_tot,),
        )

        self.list_means = means
        self.list_covs = covs
        self.blocks = blocks
        self.list_inv_covs = inv_covs
