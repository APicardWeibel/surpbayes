"""
Pseudo Gaussian distribution map, with samples drawn in the hypercube of dimension sample_size.

Function GaussHypercubeMap
"""

from typing import Optional

import numpy as np
from scipy.stats import norm
from surpbayes.misc import _get_pre_shape
from surpbayes.proba._helper import _shape_info
from surpbayes.proba.gauss.Gauss.gaussian_map import GaussianMap
from surpbayes.proba.proba_map import ProbaMap
from surpbayes.types import Samples

normal = norm(0.0, 1.0)
cdf = normal.cdf
ppf = normal.ppf
pdf = normal.pdf


def GaussHypercubeMap(
    sample_size: Optional[int] = None,
    sample_shape: Optional[tuple[int, ...]] = None,
    limits: Optional[np.ndarray] = None,
) -> ProbaMap:
    """
    Pseudo Gaussian distribution map, with samples drawn in the hypercube of dimension sample_size.

    To construct a gaussian distribution map taking values in an hypercube, limits can be passed in.
    These should be specified as an np.ndarray of shape (2, sample_size) or (2,) + sample_shape, with
    limits[:, k] specifying the interval of element k (does not have to be ordered).
    """
    sample_size, sample_shape = _shape_info(sample_size, sample_shape)

    def der_transform(xs: Samples) -> np.ndarray:
        pre_shape = _get_pre_shape(xs, sample_shape)  # type: ignore
        return np.apply_along_axis(
            np.diag, -1, pdf(xs.reshape(pre_shape + (sample_size,)))  # type: ignore
        ).reshape(
            pre_shape + sample_shape + sample_shape  # type: ignore
        )

    if limits is None:
        return GaussianMap(sample_size=sample_size).transform(
            transform=cdf, inv_transform=ppf, der_transform=der_transform
        )
    else:
        # Interpret limit as a list like of 2 elements: min/max
        limits = np.array(limits).reshape((2,) + sample_shape)

        # min/max -> mean/span
        middle = limits.mean(0)
        delta = np.abs(limits[1] - limits[0])

        def transform2(samples):
            return (samples - 0.5) * delta + middle

        def inv_transform2(samples):
            return (samples - middle) / delta + 0.5

        def der_transform2(samples):
            pre_shape = _get_pre_shape(samples, sample_shape)
            return np.full(
                pre_shape + sample_shape + sample_shape,
                np.diag(delta.flatten()).reshape(sample_shape + sample_shape),
            )

        return (
            GaussianMap(sample_size=sample_size)
            .transform(transform=cdf, inv_transform=ppf, der_transform=der_transform)
            .transform(
                transform=transform2,
                inv_transform=inv_transform2,
                der_transform=der_transform2,
            )
        )
