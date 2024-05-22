"""
Miscellanous functions used throughout the package

Functions:
    blab: wrap up for silent
    timedrun: TimeOut option for function call decorator
    interpretation: right composition decorator
    post_modif: left composition decorator
    safe_call: Evaluation of any function without failure (returns None if Exception occured)
    par_eval: Parallelisation switch for function evaluation
    num_der: numerical differentiation
    vectorize: function vectorization

Function implementations may change but input/output structure sholud remain stable.
"""

import random
import signal
import string
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
from multiprocess import Pool  # pylint:disable=E0611
from numpy.typing import ArrayLike


class ShapeError(ValueError):
    """Exception class when array shape is not as expected"""


def check_shape(xs: np.ndarray, shape_exp: tuple[int, ...]) -> None:
    """Check if xs has a given shape, raise ShapeError if not"""
    shape_obt = xs.shape
    if shape_obt != shape_exp:
        raise ShapeError(f"Shape mismatch: expected {shape_exp}, got {shape_obt}")


class ImplementationError(Exception):
    """Exception class for Implementation issues"""


def blab(silent: bool, *args, **kwargs) -> None:
    """
    Wrap up for print. If silents, does not print, else, prints.
    """
    if not silent:
        print(*args, **kwargs)


# From https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timedout after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def timedrun(max_time: int):
    """Decorator to stop function calls if they last too long
    Args:
        max_time: the maximum time for a function call before raising an exception
    Returns:
        A decorator implementing the time out mechanism on the function
    """

    def deco(function):
        def wrap(*args, **kwargs):
            with timeout(max_time):
                return function(*args, **kwargs)

        return wrap

    return deco


X1 = Any
X2 = Any
Y1 = Any
Y2 = Any


def interpretation(fun: Callable) -> Callable[[Callable[..., Y1]], Callable[..., Y1]]:
    """
    Decorator for composition on the right
    Take as argument fun, outputs a decorator transformin function g(x, ...) into g(fun(x), ...)
    Further arguments passed when constructing the decorator are passed to fun.
    """

    def deco(gfun: Callable[..., Y1], *fargs, **fkwargs) -> Callable[..., Y1]:
        def wrapper(x, *args, **kwargs):
            return gfun(fun(x, *fargs, **fkwargs), *args, **kwargs)

        return wrapper

    return deco


def post_modif(
    fun: Callable[[Y1], Y2]
) -> Callable[[Callable[..., Y1]], Callable[..., Y2]]:
    """
    Decorator for composition on the left.
    Take as argument fun, outputs a decorator transformin function g into fun(g( ...))
    Further arguments passed when constructing the decorator are passed to fun.
    """

    def deco(gfun: Callable[..., Y1], *fargs, **fkwargs) -> Callable[..., Y2]:
        def wrapper(*args, **kwargs):
            return fun(gfun(*args, **kwargs), *fargs, **fkwargs)

        return wrapper

    return deco


class SafeCallWarning(Warning):
    """Warning for safe_call context. Enables sending a warning of failure inside a safe_call
    decorated function even if UserWarning is filtered as an exception."""


# For type hints
Input = Any
Output = Any


def safe_call(fun: Callable[[Input], Output]) -> Callable[[Input], Union[None, Output]]:
    """
    Decorator to evaluate a function safely.
    If function call fails, throws a warning and returns None.

    Note:
    Decorated function can still fail IF SafeCallWarning is filtered as an error (which completely
    defeats SafeCallWarning purpose) inside fun.
    """

    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as exc:  # pylint: disable=W0703
            warnings.warn(
                f"Evaluation failed with inputs {args}, {kwargs}: {exc}",
                category=SafeCallWarning,
            )
            return None

    return wrapper


def test_fun():
    return print(__file__)


def par_eval(
    fun: Callable[..., Output], xs: Iterable[Input], parallel: bool, *args, **kwargs
) -> list[Output]:
    """
    Evaluation of a function on a list of values. If parallel is True,
    computations are parallelized using multiprocess.Pool . Else list
    comprehension is used.

    Further arguments and keyword arguments are passed to fun.
    """
    if parallel:
        loc_fun = partial(fun, *args, **kwargs)
        with Pool() as pool:  # pylint: disable=E1102
            out = pool.map(loc_fun, xs)
    else:
        out = [fun(x, *args, **kwargs) for x in xs]
    return out


def prod(x: tuple[int, ...]) -> int:
    """Minor correction to np.prod function in the case where the shape is ().
    prod is used to ensure that the product of a tuple of int outputs an int."""
    return int(np.prod(x))


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


def num_der(
    fun: Callable[[ArrayLike], ArrayLike],
    x0: ArrayLike,
    f0: Optional[ArrayLike] = None,  # pylint: disable=W0613
    rel_step: Optional[float] = None,
    parallel: bool = True,
) -> np.ndarray:
    """Return the Jacobian of a function
    If f : shape_x -> shape_y,
    the output is of shape (shape_x, shape_y)

    Arguments:
        fun: the function to derivate
        x0: the point at which to derivate the function
        f0: the value of fun at x0 (is not used since 2 point approximation of the derivative is used)
        parallel: should the evaluations of fun be parallelized
    Output:
        The approximated jacobian of fun at x0 as a np.ndarray of shape (shape_x, shape_y)
    """

    x0 = np.asarray(x0)
    shape_in = x0.shape
    x0 = x0.flatten()

    dim = prod(shape_in)
    loc_fun = interpretation(lambda x: x.reshape(shape_in))(fun)

    if rel_step is None:
        rel_step = float((np.finfo(x0.dtype).eps) ** (1 / 3))

    to_evaluate = np.full((2 * dim, dim), x0)

    delta_x = np.maximum(1.0, x0) * rel_step
    add_matrix = np.diag(delta_x)
    to_evaluate[::2] = to_evaluate[::2] + add_matrix

    to_evaluate[1::2] = to_evaluate[1::2] - add_matrix

    evals = np.array(par_eval(loc_fun, to_evaluate, parallel=parallel))

    der = evals[::2] - evals[1::2]

    for i, d_x in enumerate(delta_x):
        der[i] = der[i] / (2 * d_x)
    shape_out = der[0].shape
    return der.reshape(shape_in + shape_out)


def safe_inverse_ps_matrix(matrix: np.ndarray, eps: float = 10**-6) -> np.ndarray:
    """Compute the inverse of symmetric positive definite matrix.
    Correct np.linalg.inv implementation when matrix has bad conditionning number.

    Inversion of eigenvalues < eps will be heavily perturbed.
    This function is still work in progress
    """

    return np.linalg.inv(0.5 * (matrix + matrix.T) + eps * np.eye(len(matrix)))


def vectorize(
    fun: Callable,
    input_shape: tuple[int, ...],
    convert_input: bool = True,
    parallel: bool = True,
) -> Callable:
    """For a function fun which takes as input np.ndarray of shape input_shape and outputs
    arrays of shape output_shape, outputs the vectorized function which takes as input np.ndarray
    of shape (pre_shape, input_shape) and outputs np.ndarrat of shape (pre_shape, output_shape)
    """
    d = len(input_shape)

    def new_fun(xs) -> np.ndarray:
        if convert_input:
            xs = np.asarray(xs)
        pre_shape = xs.shape[:-d]
        xs.reshape(
            (
                (
                    prod(
                        pre_shape,
                    ),
                )
                + input_shape
            )
        )
        out = np.array(par_eval(fun, xs, parallel=parallel))
        out.reshape(pre_shape + out.shape[1:])
        return out

    return new_fun


def random_name(k: int = 30):
    """Generate a random name for a temporary file/folder"""
    return "".join(random.choices(string.ascii_lowercase, k=k))


def check_set_equal(iter_1, iter_2):
    """Check if two set like object have identical keys. If not, raise a ValueError"""
    err_msgs = []
    extra_1 = set(iter_1).difference(iter_2)
    if extra_1:
        err_msgs.append(f"Index 1 contains extra keys {extra_1}")
    extra_2 = set(iter_2).difference(iter_1)
    if extra_2:
        err_msgs.append(f"Index 2 contains extra keys {extra_2}")

    err_msg = "\nFurthermore:".join(err_msgs)

    if err_msg:
        raise ValueError(err_msg)
