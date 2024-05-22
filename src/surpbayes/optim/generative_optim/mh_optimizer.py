"""
Optimisation using modified Metropolis Hastings algorithm
"""

from typing import Callable, Optional, Union

import numpy as np

from surpbayes.accu_xy import AccuSampleVal
from surpbayes.misc import blab, check_shape
from surpbayes.optim.generative_optim.gen_optim import GenOptimizer


class MHOptimizer(GenOptimizer):
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        param_ini: np.ndarray,
        # Termination criteria
        chain_length: int = 100,
        xtol: float = 10 ** (-8),
        ftol: float = 10 ** (-8),
        delta_conv_check: int = 10,
        # Function evaluation
        parallel: bool = True,
        vectorized: bool = False,
        # MH specific
        per_step: Optional[Union[int, list[int]]] = None,
        prev_eval: Optional[AccuSampleVal] = None,
        prop_cov: Optional[np.ndarray] = None,
        radius_ini: Optional[float] = 1.0,
        radius_factor: float = 0.7,
        no_change_max: int = 10,
        # Printing
        silent=False,
        print_rec: int = 5,
        **kwargs,
    ):

        super().__init__(
            fun=fun,
            param_ini=param_ini,
            chain_length=chain_length,
            xtol=xtol,
            ftol=ftol,
            delta_conv_check=delta_conv_check,
            per_step=per_step,
            prev_eval=prev_eval,
            parallel=parallel,
            vectorized=vectorized,
            silent=silent,
            print_rec=print_rec,
            **kwargs,
        )

        self.radius = radius_ini
        self.radius_factor = radius_factor
        self.no_change_max = no_change_max

        if prop_cov is None:
            self.cov = np.eye(self.dim_par)
        else:
            self.cov = prop_cov

        check_shape(self.cov, (self.dim_par, self.dim_par))

        # To gain time, random proposals are drawned at the beginning
        self.proposals = np.random.multivariate_normal(
            mean=np.zeros(self.dim_par), cov=self.cov, size=sum(self.per_step)
        )

        self.per_step_cuts = np.concatenate([[0], np.cumsum(self.per_step)])

        self.count_fails = 0

    def update(self):
        self.msg_begin_step()

        # Get parameters to evaluate
        prop_mod = self.proposals[
            self.per_step_cuts[self.count] : self.per_step_cuts[self.count + 1]
        ]
        draw = self.param + self.radius * prop_mod
        evals = self.mfun(draw)

        self.accu.add(sample=draw, vals=evals)

        if np.all(evals >= self.score):
            conseq_fails = self.per_step[self.count]
        else:
            conseq_fails = np.argmin(evals >= self.score)
        conseq_fails = self.count_fails + conseq_fails

        arg_min = np.argmin(evals)

        if evals[arg_min] < self.score:
            self.param = draw[arg_min]
            self.score = evals[arg_min]
            self.count_fails = 0
        else:
            # Update the successive failure (i.e. no score decrease) count
            self.count_fails += self.per_step[self.count]

        if conseq_fails < self.no_change_max:
            # Reset the successive failure (i.e. no score decrease) count and update radius
            self.count_fails = 0
            self.radius = self.radius * self.radius_factor

            blab(self.silent, f"New proposal radius: {self.radius}")

        # Update step count
        self.count = self.count + 1
        # Add best parameter found so far to accu
        self.hist_log.add1(self.param, self.score)

        self.check_convergence()
        self.msg_end_step()
