"""
Variant of gaussian hypercube for those wanting to define distributions on the standard L^2 ball
"""

import numpy as np
from scipy.stats import chi2

from surpbayes.types import Samples
from surpbayes.misc import prod
from surpbayes.proba.gauss.Gauss.gaussian_map import GaussianMap
from surpbayes.proba.proba_map import ProbaMap


def GaussHyperballMap(sample_size: int) -> ProbaMap:
    """
    Pseudo Gaussian distribution map, with samples drawn in the hypercube of dimension sample_size
    """
    chi2_to_use = chi2(sample_size)
    pdf = chi2_to_use.pdf
    ppf = chi2_to_use.ppf
    cdf = chi2_to_use.cdf

    def transform(xs: Samples) -> Samples:
        pre_shape = xs.shape[:-1]
        xs = xs.reshape((prod(pre_shape), xs.shape[-1]))

        norm2_x = (xs**2).sum(-1)
        new_norm = cdf(norm2_x) ** (1 / sample_size)
        return ((xs.T * new_norm / np.sqrt(norm2_x)).T).reshape(
            pre_shape + (xs.shape[-1],)
        )

    def inv_transform(ys: Samples) -> Samples:
        pre_shape = ys.shape[:-1]
        ys = ys.reshape((prod(pre_shape)), ys.shape[-1])
        norm_y = np.sqrt((ys**2).sum(-1))  # = new_norm in prev
        norm2_x = ppf(norm_y**sample_size)
        return (ys.T * np.sqrt(norm2_x)).T.reshape(pre_shape + (ys.shape[-1],))

    def der_transform(xs: Samples) -> np.ndarray:
        """TO DO: Compute the limit when x -> 0 (seems to be of form alpha Id"""
        pre_shape = xs.shape[:-1]
        xs = xs.reshape((prod(pre_shape), xs.shape[-1]))

        norm2_x, der_norm2_x = (xs**2).sum(-1), 2 * xs
        norm_x = np.sqrt(norm2_x)  # n_samples
        normed_x = (xs.T / norm_x).T  # shape (n_samples, n_dim)

        der_normed_x = (
            (
                np.eye(sample_size)
                - np.apply_along_axis(lambda x: np.outer(x, x), -1, normed_x)
            ).T
            / norm_x
        ).T  # Shape n_samples, n_dim, n_dim

        cdf_norm_x, der_cdf_norm_x = cdf(norm2_x), pdf(
            norm2_x
        )  # Shape n_samples / Shape n_samples

        new_norm, der_new_norm = (
            cdf_norm_x ** (1 / sample_size),  # n_samples
            (
                der_norm2_x.T  # (n_dim, n_samples)
                * der_cdf_norm_x  # n_samples
                * (cdf_norm_x ** (1 / sample_size - 1))  # n_samples
                / sample_size
            ),  # int
        )  # Shape n_samples/ (n_dim, n_samples)

        # normed_x: shaped (n_samples, n_dim)
        # der_new_norm: shapes n_dim, n_samples
        normed_x = normed_x.reshape((prod(pre_shape), sample_size))
        der_new_norm = der_new_norm.reshape(((sample_size, prod(pre_shape))))
        to_add = np.array(
            [np.outer(normed_x[:, i], der_new_norm[i]) for i in range(prod(pre_shape))]
        ).reshape(pre_shape + (sample_size, sample_size))
        return (new_norm * der_normed_x.T).T + to_add

    return GaussianMap(sample_size=sample_size).transform(
        transform=transform, inv_transform=inv_transform, der_transform=der_transform
    )
