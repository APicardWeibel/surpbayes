"""Miscallenous helper function for Gaussian distributions/Maps"""

import warnings
from typing import Optional

import numpy as np

from surpbayes.misc import ShapeError, prod
from surpbayes.proba.warnings import ShapeWarning


def _shape_info(
    sample_size: Optional[int] = None, sample_shape: Optional[tuple[int, ...]] = None
) -> tuple[int, tuple[int, ...]]:
    """
    Check and format shape information from potentially incomplete information
    """
    if (sample_size is None) and (sample_shape is None):
        raise ValueError("Either 'sample_size' or 'sample_shape' must be specified.")

    if sample_shape is None:
        sample_shape = (sample_size,)  # type: ignore

    elif sample_size is None:
        sample_size = prod(
            sample_shape
        )  # Define if sample_size is missing/Force coherence if both are specified

    elif sample_size != prod(sample_shape):  # type: ignore
        warnings.warn(  # type: ignore
            message=f"'sample_size' {sample_size} and 'sample_shape' {sample_shape} arguments are incoherent. Using 'sample_shape' information",
            cateogry=ShapeWarning,
        )
        sample_size = prod(sample_shape)

    return sample_size, sample_shape  # type: ignore


def _get_pre_shape(xs: np.ndarray, exp_shape: tuple[int, ...]) -> tuple[int, ...]:
    if exp_shape == ():
        return xs.shape
    n_dim = len(exp_shape)
    tot_shape = xs.shape

    if len(tot_shape) < n_dim:
        raise ShapeError("Shape of input array is not compliant with expected shape")

    if tot_shape[-n_dim:] != exp_shape:
        raise ShapeError("Shape of input array is not compliant with expected shape")

    return tot_shape[:-n_dim]
