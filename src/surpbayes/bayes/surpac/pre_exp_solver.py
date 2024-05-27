"""Submodule for SurPACSolver subclass for PreExepFamily"""

from typing import Callable, Optional, Union

import numpy as np

from surpbayes.types import ProbaParam, SamplePoint
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.bayes.surpac.surpac_solver import (
    SurPACSolver,
)
from surpbayes.misc import blab
from surpbayes.optim import dichoto
from surpbayes.proba import PreExpFamily, RenormError

# from multiprocess import Pool  # pylint: disable=E0611


def _solve_in_kl(
    proba_map: PreExpFamily,
    post_param: ProbaParam,
    t_post: np.ndarray,
    direction: np.ndarray,
    kl_max: float,
    alpha_max: float,
    y_pres: Optional[float] = None,
    x_pres: Optional[float] = None,
    m_max: int = 100,
) -> float:
    """
    Solve the equation "kl(new_post_param(alpha), post_param) = kl_max"
    where new_post_param(alpha) = T_to_param(param_to_T(post_param) + alpha * direction).

    The equation is solved through a dichotomy solver. It is not garanteed that the smallest
    solution will be returned, as the function might not be non decreasing. Experimentations so
    far only obtained increasing function (due to the T transform).
    """

    def loc_fun(alpha):
        # Compute kl(new_post_param(alpha), post_param) safely.
        # If for value alpha, new_post_param does not exist (raise RenormError),
        # returns infinity (this prevents failure using dichotomy)
        try:
            new_t_post = t_post + alpha * direction
            new_post_param = proba_map.T_to_param(new_t_post)
        except RenormError:
            return np.inf
        return proba_map.kl(new_post_param, post_param)

    # implement solver for loc_fun = kl_max, assuming loc_fun is increasing in alpha.
    # Use dichotomy, take lower value

    # Check if condition is not already met
    if loc_fun(alpha_max) < kl_max:
        return alpha_max

    # Use dichotomy as alpha_max is too high.
    return dichoto(
        loc_fun,
        kl_max,
        0,
        alpha_max,
        increasing=True,
        y_pres=y_pres,
        x_pres=x_pres,
        m_max=m_max,
    )[0]


class PreExpSPACS(SurPACSolver):
    """Bayesian Optimisation using Score approximation routine for PreExpFamily Maps

    Methods for the computation of the update correction are adapted ('get_t_dir',
    'get_updt_speed', 'updt_post_par')
    """

    def __init__(
        self,
        fun: Callable[[SamplePoint], float],
        proba_map: PreExpFamily,
        prior_param: Optional[ProbaParam] = None,
        post_param: Optional[ProbaParam] = None,
        temperature: float = 1.0,
        prev_eval: Optional[AccuSampleVal] = None,
        n_estim_weights: int = 10**5,
        kl_max: float = 10.0,
        dampen: float = 0.1,
        chain_length: int = 10,
        n_max_eval: Optional[int] = None,
        per_step: Union[int, list[int]] = 100,
        kltol: float = 10**-8,
        # arguments for dichotomy (prevent KL(post_{n+1}, post_n)> kl_max)
        y_pres: Optional[float] = None,
        x_pres: Optional[float] = None,
        m_max: int = 100,
        # For function evaluation
        parallel: bool = True,
        vectorized: bool = False,
        print_rec: int = 1,
        silent: bool = False,
        **kwargs,
    ):
        super().__init__(
            fun=fun,
            proba_map=proba_map,
            prior_param=prior_param,
            post_param=post_param,
            temperature=temperature,
            prev_eval=prev_eval,
            n_estim_weights=n_estim_weights,
            kl_max=kl_max,
            dampen=dampen,
            chain_length=chain_length,
            n_max_eval=n_max_eval,
            per_step=per_step,
            kltol=kltol,
            y_pres=y_pres,
            x_pres=x_pres,
            m_max=m_max,
            parallel=parallel,
            vectorized=vectorized,
            print_rec=print_rec,
            silent=silent,
            **kwargs,
        )

        self.t_post = self.proba_map.param_to_T(self._post_param)
        self.t_prior = self.proba_map.param_to_T(self.prior_param)

    def msg_begin_calib(self) -> None:
        blab(
            self.silent,
            " ".join(
                [
                    "Starting Bayesian calibration",
                    "(Score Approximation routine,",
                    "Pre Exponential Family variant)",
                ]
            ),
        )

    def get_t_dir(self, t_score: np.ndarray) -> np.ndarray:
        """Computation of direction in natural parametrisation"""
        return self.t_prior - self.inv_temp * t_score - self.t_post

    def get_updt_speed(self, t_dir: np.ndarray) -> np.ndarray:
        """Update speed for natural parametrisation. Enforce constraints that
        KL(new post, current post) < kl_max
        and the factor is lower than 1- dampen
        """

        return _solve_in_kl(
            proba_map=self.proba_map,
            post_param=self._post_param,
            t_post=self.t_post,
            direction=t_dir,
            kl_max=self.kl_max,
            alpha_max=(1 - self.dampen),
            y_pres=self.y_pres,
            x_pres=self.x_pres,
            m_max=self.m_max,
        )

    def updt_post_par(self, t_dir: np.ndarray, alpha: float):
        self.t_post = self.t_post + alpha * t_dir
        new_post_param = self.proba_map.T_to_param(self.t_post)
        self.check_convergence(new_post_param)
        self._post_param = new_post_param
        self._post = self.proba_map(self._post_param)
