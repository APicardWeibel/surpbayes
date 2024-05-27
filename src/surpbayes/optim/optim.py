"""Optimisation function 'optim'
"""
from typing import Union

import numpy as np
from surpbayes.optim.generative_optim.cma_optimizer import CMAOptimizer
from surpbayes.optim.generative_optim.mh_optimizer import MHOptimizer
from surpbayes.optim.optim_result import OptimResult
from surpbayes.optim.optimizer import Optimizer


def optim(
    fun,
    param_ini: np.ndarray,
    optimizer: Union[str, Optimizer],
    parallel: bool = True,
    vectorized: bool = False,
    **kwargs,
) -> OptimResult:
    """Optimisation function

    Instantiate an optimizer, runs optimization process and return an OptimResult

    Args:
        fun: (Callable, returns float) the function to be minimized
        param_ini: (Array like) initial guess for the minima of the function
        optimizer: (str or Optimizer class) the optimisation method to be used (either 'CMA', 'MH'
            or a custom optimizer class)
        parallel: whether 'fun' calls should be parallelized (through multiprocess)
        vectorized: whether 'fun' is vectorized (if True, 'parallel' is disregarded)
    Further kwargs are passed to the optimizer instance (and then to 'fun' if not caught by the
    optimizer initialization)

    Outputs:
        An OptimResult object (main attributes: "opti_param", "opti_score", "converged")

    Note on Optimizer class:
    A subclass of Optimizer must follow the following guidelines:
    - optimisation is runned through "optimize" method
    - the result of the optimisation process is obtained by "process_result" method

    See documentation of Optimizer class for further details
    -----------------------------------------------------------------------------------------------
    Details on further arguments

    For CMA-ES routine (optmizer == 'CMA')
        chain_length: maximum length of the optimisation chain
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
        print_rec: specify how often should there be prints.
            Information on the optimisation is printed every print_rec steps if silent is False
        silent: should there be any prints?

    For Metropolis Hastings routine (optimizer == 'MH')
        xtol, ftol: criteria for termination (converged when one of the criteria is met)
        per_step: Number of samples generated and evaluated at each step
        prop_cov: Initial covariance structure on parameters
        radius_ini: Multiplication factor for proposals
            (amounts to a initial covariance of (radius_ini ** 2) * cov_ini)
        radius_factor: contraction factor for the covariance when failing to find lower score.
        no_change_max: Number of samples drawn without finding a parameter achieving lower
            score before the covariance is contracted by radius_factor ** 2
        parallel: should the calls to the score be parallelized (during each step)
        vectorized: is the score function assumed to be vectorized? Default is False. If True,
            parallel is disregarded
        print_rec: specify how often should there be prints.
            Information on the optimisation is printed every print_rec steps if silent is False
        silent: should there be any prints?
    """
    if isinstance(optimizer, str):
        if optimizer == "CMA":
            optim = CMAOptimizer(
                fun=fun,
                param_ini=param_ini,
                parallel=parallel,
                vectorized=vectorized,
                **kwargs,
            )
        elif optimizer == "MH":
            optim = MHOptimizer(
                fun=fun,
                param_ini=param_ini,
                parallel=parallel,
                vectorized=vectorized,
                **kwargs,
            )
        else:
            raise ValueError(f"'optimizer' {optimizer} is not recognized")

    else:
        optim = optimizer(
            param_ini=param_ini, parallel=parallel, vectorized=vectorized, **kwargs
        )

    try:
        optim.optimize()
        return optim.process_result()

    except Exception as exc:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"'optimizer' {optimizer} is not of valid type") from exc
        raise exc
