"""
Optimisation using CMA-ES routine
See demo or https://doi.org/10.48550/arXiv.1604.00772 for description of CMA-ES

Implementation note:
    - 'par_eval' is not used here since opening/closing pools of workers takes unnecessary time.

Future:
    - New points could be more smartly generated
"""

import os
from typing import Callable, Optional, Union

import numpy as np
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.misc import blab, check_shape, prod
from surpbayes.optim.generative_optim.gen_optim import GenOptimizer
from surpbayes.optim.optim_result import OptimResult
from surpbayes.types import Samples


class OptimResultCMA(OptimResult):
    """
    Subclass of OptimResult for output of CMA-ES optimisaiton algorithm

    This class functions as an organized storage of optimisation related variables. These include
    - opti_param, the parameter returned by the optimisation routine
    - converged, whether the optimisation routine assumes convergence
    - opti_score, the score achieved by the optimisation routine (Optional)
    - hist_param, the list of parameters in the optimisation route (Optional)
    - hist_score, the scores of the parameters in hist_param (Optional)
    - full_evals, the full evaluations of x_i, S(x_i) generated during optimisation (Optional)
    - hyperparams, the hyperparameters used for the optimisation procedure (Optional)
    - hist_cov, the list of covariance used during the CMA training algorithm
    """

    class_name = "OptimResultCMA"

    def __init__(
        self,
        opti_param,
        converged: bool,
        opti_score: Optional[float],
        hist_param: Optional[np.ndarray],
        hist_score: Optional[np.ndarray],
        hist_cov: Optional[np.ndarray],
        full_evals: Optional[AccuSampleVal],
        hyperparams: Optional[dict] = None,
    ):
        super().__init__(
            opti_param=opti_param,
            converged=converged,
            opti_score=opti_score,
            hist_param=hist_param,
            hist_score=hist_score,
            hyperparams=hyperparams,
        )
        self._full_evals = full_evals
        self._hist_cov = hist_cov

    @property
    def full_evals(self):
        return self._full_evals

    @property
    def hist_cov(self):
        return self._hist_cov

    def convert(self, fun: Callable, vectorized: bool = False, parallel: bool = False):
        super().convert(fun=fun, vectorized=vectorized, parallel=parallel)

        if self._full_evals is not None:
            self._full_evals.convert(fun, vectorized, parallel)

    def save(self, name: str, path: str = ".", overwrite: bool = True) -> str:
        """Save 'OptimResultCMA' object to folder 'name' in 'path'."""

        # Save all attributes in parent class
        acc_path = super().save(name, path, overwrite)

        # Save additional attribute hist_cov
        hicov_arr = np.array(np.array(self.hist_cov))
        np.savetxt(
            os.path.join(acc_path, "hist_cov.csv"),
            hicov_arr.reshape((prod(hicov_arr.shape[:2]), hicov_arr.shape[2])),
        )

        # Save additional attribute full_evals
        if self.full_evals is not None:
            self.full_evals.save("full_evals", path=acc_path, overwrite=overwrite)

        return acc_path


class CMAOptimizer(GenOptimizer):
    """Optimisation algorithm using Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    algorithm. CMA-ES algorithm is described in https://doi.org/10.48550/arXiv.1604.00772
    The implementation is original.

    Args:
        param_ini: initial mean parameter
        score: scoring function, to be minimized
        chain_length: maximum length of the chain
        xtol, ftol: criteria for termination (converged when one of the criteria is met)
        cov_ini: Initial covariance structure on parameters
        radius_ini: Multiplication factor for proposals
            (amounts to a initial covariance of (radius_ini ** 2) * cov_ini)
        per_step: Number of samples generated and evaluated at each step
        no_change_max: Number of samples drawn without finding a parameter achieving lower
            score before the covariance is contracted by radius_factor ** 2
        radius_factor: contraction factor for the covariance when failing to find lower score.
        cov_updt_speed: control on the speed of the covariance update. The covariance at time t+1
            is (1- cov_updt_speed) * cov_t + cov_updt_speed * cov_updt .
            Default is 0.1
        keep_frac: fraction of good draws used to define the update of the covariance.
            Default is 0.25
        n_speed_comp: number of steps used to compute the current average y decrease speed.
            Used for termination. Default is 30.
        parallel: should the calls to the score be parallelized (during each step)
        vectorized: is the score function assumed to be vectorized? Default is False. If True,
            parallel is disregarded
        print_rec: specify how often should there be prints.
            Information on the optimisation is printed every print_rec steps if silent is False
        silent: should there be any prints?

    Outputs:
        An OptimResultCMA object, inherited from OptimResults, with attributes
            opti_param, opti_score, converged, hist_param, hist_scores, hist_cov.
    """

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
        # CMA-ES specific
        per_step: Optional[Union[int, list[int]]] = None,
        prev_eval: Optional[AccuSampleVal] = None,
        cov_ini: Optional[np.ndarray] = None,
        radius_ini: Optional[float] = 1.0,
        no_change_max: int = 10,
        radius_factor: Optional[float] = None,
        cov_updt_speed: float = 0.1,
        keep_frac: float = 0.25,
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

        self.no_change_max = no_change_max
        self.cov_updt_speed = cov_updt_speed

        # Choose radius_factor from cov_updt_speed if missing
        if radius_factor is None:
            self.radius_factor = np.sqrt(1 - self.cov_updt_speed)
        else:
            self.radius_factor = radius_factor

        # Construct initial covariance
        if cov_ini is None:
            self.cov = radius_ini**2 * np.eye(self.dim_par)
        else:
            self.cov = radius_ini**2 * cov_ini

        check_shape(self.cov, (self.dim_par, self.dim_par))

        # ---------- Set up accus ----------
        self.accu_cov = np.zeros((chain_length + 1,) + self.cov.shape)
        self.accu_cov[0] = self.cov

        self.keep_frac = keep_frac

        # Other arguments
        self.max_keep = [max(int(x * keep_frac), 1) for x in self.per_step]
        self.count_fails = 0

    def gen(self, n) -> Samples:
        """Generate samples from current state - random draws using np.random"""
        return np.random.multivariate_normal(self.param, self.cov, n).reshape(
            (n,) + self.par_shape
        )

    def update(self) -> None:
        """Optimisation step"""
        self.msg_begin_step()
        # Infer step hyperparameters
        per_step = self.per_step[self.count]
        max_keep = self.max_keep[self.count]

        # Sample from current distribution
        draw = self.gen(per_step)
        evals = self.mfun(draw)

        self.accu.add(sample=draw, vals=evals)

        # Compute consecutive failures (for radius update)
        if np.all(evals >= self.score):
            conseq_fails = per_step
        else:
            conseq_fails = np.argmin(evals >= self.score)
        conseq_fails = self.count_fails + conseq_fails

        # Sort scores
        sorter = np.argsort(evals)
        keep_draw = draw[sorter][:max_keep]
        evals = evals[sorter][:max_keep]

        if evals[0] <= self.score:
            # Case where a better score was found
            # Consider only parameters achieving a better score than previously found
            keep_draw = keep_draw[evals <= self.score]
            evals = evals[evals <= self.score]

            # The new center of the distribution is the parameter achieving the best score so far
            new_param = keep_draw[0]
            new_score = evals[0]

            # Compute soft covariance update
            # Compute pseudo covariance of sample, using the previous mean
            # This helps preferentially drawing along "new_param - start_param" axis
            add_cov = np.tensordot(
                keep_draw - self.param, keep_draw - self.param, (0, 0)
            ) / len(keep_draw)

            # Update current score, param and covariance
            self.score = new_score
            self.param = new_param
            self.cov = (1 - self.cov_updt_speed) * self.cov + (
                self.cov_updt_speed
            ) * add_cov

            # Update the successive failure (i.e. no score decrease) count
            self.count_fails = 0
        else:
            # Update the successive failure (i.e. no score decrease) count
            self.count_fails += per_step

        if conseq_fails >= self.no_change_max:
            blab(self.silent, "Updating covariance radius")
            self.cov = (self.radius_factor**2) * self.cov
            self.count_fails = 0

        # Update step count
        self.count += 1

        # Add best parameter found so far to accu
        self.hist_log.add1(self.param, self.score)
        self.accu_cov[self.count] = self.cov

        self.check_convergence()
        self.msg_end_step()

    def get_hyperparams(self) -> dict:
        """Process hyperparameters of CMA-ES algorithm to a dictionnary"""
        dico = super().get_hyperparams()
        dico.update(
            {
                "per_step": self.per_step,
                "no_change_max": self.no_change_max,
                "keep_frac": self.keep_frac,
                "cov_updt_speed": self.cov_updt_speed,
                "radius_factor": self.radius_factor,
            }
        )

    def process_result(self) -> OptimResultCMA:
        """Process optimisation results to an OptimResultCMA object"""
        return OptimResultCMA(
            opti_param=self.param,
            converged=self.converged,
            opti_score=self.score,
            hist_param=self.hist_log.sample(),
            hist_score=self.hist_log.vals(),
            hist_cov=self.accu_cov[: (self.count + 1)],
            full_evals=self.accu,
            hyperparams=self.get_hyperparams(),
        )
