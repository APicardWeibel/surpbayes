""" Bound support of functions for optimization """

import warnings

import numpy as np

from surpbayes.misc import _get_pre_shape, prod


def bound_func(
    fun,
    lower_bounds=-np.inf,
    upper_bounds=np.inf,
    fail_return=np.inf,
    input_shape=None,
    vectorized=False,
):
    """Pseudo decorator to impose bounds during optimisation"""
    if (lower_bounds is -np.inf) and (upper_bounds is np.inf):
        return fun

    if vectorized:
        axes = tuple(i for i in range(-len(input_shape), 0))

        if input_shape is None:
            warnings.warn(
                "bound_support can not be used if 'input_shape' is not specified"
            )
        else:

            def wrapper(x: np.ndarray, *args, **kwargs):
                failed = np.any(x < lower_bounds, axis=axes) | np.any(
                    x > upper_bounds, axis=axes
                )
                pre_shape = _get_pre_shape(x, input_shape)
                out = fun(x.reshape((prod(pre_shape),) + input_shape), *args, **kwargs)
                out[failed.flatten()] = fail_return
                return out.reshape(pre_shape)

            return wrapper

    def wrapper(x: np.ndarray, *args, **kwargs):  # pylint: disable=E0102
        if np.any(x < lower_bounds) or np.any(x > upper_bounds):
            return fail_return
        return fun(x, *args, **kwargs)

    return wrapper
