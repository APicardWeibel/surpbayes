"""
The SVGD routine is used to optimize the position of the last points, the other remaining
fixed, hence Partial. It is written assuming Gaussian distributions, though general distributions
could be used as well (need to encode the log density derivative with respect to the sample point).
The kernel in the general case also needs to be adapted (no inverse covariance accessible).

Part of the implementation is adapted from https://random-walks.org/content/misc/svgd/svgd.html,
with the formula for the update rule corrected.
"""

from typing import Callable

import numpy as np
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.misc import par_eval, prod
from surpbayes.proba import ExponentialFamily, Gaussian
from surpbayes.types import ProbaParam, Samples


def eq(inv_cov):
    def kernel(x, x_) -> tuple[np.ndarray, np.ndarray]:
        """
        x is shaped (n_x, k)
        x_ is shaped (n_x2, k)
        """
        diff = x[:, np.newaxis] - x_[np.newaxis, :]  # shaped (n_x, n_x_, p)
        quad = (diff * (diff @ inv_cov)).sum(-1)
        ker_res = np.exp(-0.5 * quad)  # matrix of size (n_x, n_x_)
        return ker_res, ker_res[:, :, np.newaxis] * (
            diff @ inv_cov
        )  # shaped (n_x, n_x_, p)

    return kernel


def get_der_x_log_dens(prob_param: ProbaParam, exp_family: ExponentialFamily):
    """
    Derivative of log_density with respect to $x$ for a distribution in an exponential family.
    """
    dims_log_dens_help = tuple(-i - 1 for i in range(len(exp_family.proba_param_shape)))
    if exp_family.der_T is None:
        raise ValueError("Required der_T attribute")

    if exp_family.der_h is None:
        # assuming that h is constant for Lebesgue measure

        def fun(xs: Samples):
            return (exp_family.der_T(xs) * prob_param).sum(dims_log_dens_help)  # type: ignore

    else:

        def fun(xs: Samples):
            return (exp_family.der_T(xs) * prob_param).sum(  # type: ignore
                dims_log_dens_help
            ) + exp_family.der_T(  # type: ignore
                xs
            )  #

        return fun


def _update_rule(
    xs: Samples,
    der_x_log_dens: Callable[[Samples], np.ndarray],
    kernel: Callable[[Samples, Samples], tuple[np.ndarray, np.ndarray]],
    retain: int = 0,
):
    ks, ks_der = kernel(xs[retain:], xs)  # shaped (n_x - retain, n_x)

    glogs = der_x_log_dens(xs)  # shaped (n_x, p)
    f_el = (ks[:, :, np.newaxis] * glogs).sum(1)  # shaped (n_x - retain, p)

    s_el = ks_der.sum(1)

    return (f_el + s_el) * (1 / len(xs))


def partial_svgd(
    xs: Samples,
    prob_param: ProbaParam,
    exp_family: ExponentialFamily,
    kernel: Callable[[Samples, Samples], tuple[np.ndarray, np.ndarray]],
    retain: int = 0,
    epsilon=0.01,
    n_repeat=100,
):
    """
    Stein Variational gradient descent for a distribution in an exponential family.
    """
    xs = xs.copy()
    der_x_log_dens = get_der_x_log_dens(prob_param=prob_param, exp_family=exp_family)

    for _ in range(n_repeat):
        xs[retain:] = xs[retain:] + epsilon * _update_rule(
            xs, der_x_log_dens, kernel, retain
        )
    return xs


def enrich_sample_svgd(
    prob_param: ProbaParam,
    exp_family: ExponentialFamily,
    accu_sample: AccuSampleVal,
    score: Callable,
    n_new: int,
    kernel=None,
    epsilon: float = 0.01,
    n_repeat: int = 100,
    vectorized: bool = False,
    parallel: bool = True,
):
    """Enriching samples using SVGD algorithm"""

    if kernel is None:
        kernel = eq(np.eye(prod(exp_family.sample_shape)))

    sample = accu_sample.sample()
    retain = accu_sample.n_filled
    proba = exp_family(prob_param)
    new_points = proba(n_new)

    xs = np.concatenate([sample, new_points], axis=0)
    new_points = partial_svgd(
        xs=xs,
        prob_param=prob_param,
        exp_family=exp_family,
        kernel=kernel,
        retain=retain,
        epsilon=epsilon,
        n_repeat=n_repeat,
    )
    if vectorized:
        values = score(new_points)
    else:
        values = par_eval(score, new_points, parallel)

    accu_sample.add(new_points, values)  # in place


# Special case for gaussian distributions
def get_der_x_log_dens_gauss(proba: Gaussian):
    """Derivative with respect to x of the log density for gaussian distributions"""

    def function(xs: Samples):
        return -(xs - proba.means) @ proba.inv_cov

    return function


def partial_svgd_gauss(
    xs: Samples, proba: Gaussian, retain: int = 0, epsilon=0.01, n_repeat=100
):
    """Stein variational gradient descent for gaussian distribution"""
    xs = xs.copy()
    kernel = eq(proba.inv_cov)
    der_x_log_dens = get_der_x_log_dens_gauss(proba)
    for _ in range(n_repeat):
        xs[retain:] = xs[retain:] + epsilon * _update_rule(
            xs, der_x_log_dens, kernel, retain
        )
    return xs


def enrich_sample_svgd_gauss(
    proba: Gaussian,
    accu_sample: AccuSampleVal,
    score: Callable,
    n_new: int,
    epsilon: float = 0.01,
    n_repeat: int = 100,
    vectorized: bool = False,
    parallel: bool = True,
):
    """
    Enriching samples using SVGD algorithm. Gaussian case variant.

    Contrary to the standard case, the kernel is inferred from the covariance matrix.
    """
    sample = accu_sample.sample()
    retain = accu_sample.n_filled
    new_points = proba(n_new)

    xs = np.concatenate([sample, new_points], axis=0)
    new_points = partial_svgd_gauss(
        xs=xs, proba=proba, retain=retain, epsilon=epsilon, n_repeat=n_repeat
    )
    if vectorized:
        values = score(new_points)
    else:
        values = par_eval(score, new_points, parallel)

    accu_sample.add(new_points, values)  # in place
