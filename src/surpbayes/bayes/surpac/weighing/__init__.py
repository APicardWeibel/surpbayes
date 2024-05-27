r"""
Weighting step for score_approximation Bayesian inference framework.

Sub module for weighing a sample of points in order to approximate a distribution.

Methods in the Bayesian calibration module rely on estimation of integrals with respect to
variable distributions :math:`\pi(\theta)`.

These integrals typically involve a function f which is costly to evaluate. Previous evaluations
of f at a fixed sample :math:`(x_i)` is available.

The goal of the present module is to compute weights :math:`(\omega_i)`, such that
..math::
    \sum g(x_i)\omega_i \simeq  \pi(\theta)[g]
for all function g.
"""

from .monte_carlo import (get_weights_mc, get_weights_mc_approx,
                          get_weights_mc_approx_gauss, get_weights_mc_gauss)
