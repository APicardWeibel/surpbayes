"""
Function form for PAC Bayes minimisation, with solver inference.
"""


from typing import Callable, Optional, Type, Union

import numpy as np
from surpbayes.bayes.bayes_solver import BayesSolver
from surpbayes.bayes.gradient_based import (GradientBasedBayesSolver,
                                            KNNBayesSolver)
from surpbayes.bayes.optim_result_bayes import OptimResultBayes
from surpbayes.bayes.surpac import (GaussianSPACS, PreExpSPACS,
                                          SurPACSolver)
from surpbayes.proba import ProbaMap
from surpbayes.types import ProbaParam

set_pac_bayes_solver = {
    "corr_weights",
    "knn",
    "SurPAC-CE", 
    "score_approx", # old name for SurPAC-CE, still used as reference until mod
    "score_approx_gauss",
    "score_approx_pre_exp",
    "score_approx_exp",
}


def infer_pb_routine(
    proba_map: ProbaMap, pac_bayes_solver: Optional[Union[str, Type[BayesSolver]]] = None
) -> Type[BayesSolver]:
    """Infer which pac_bayes_solver from 'proba_map' and 'pac_bayes_solver' arguments.

    Check Coherence between 'pac_bayes_solver' and 'proba_map'.

    Rules:
        If None, defaults to 'corr_weights' for generic distribution and the adequate
        'score_approx' routines for ExponentialFamily, PreExpFamily, and Gaussian related
        distributions.
        If 'score_approx', checks the appropriate version of 'score_approx' depending on the
        'proba_map' passed.
    """
    if (pac_bayes_solver is None) or (pac_bayes_solver == "score_approx") or (pac_bayes_solver == "SurPAC-CE"):
        if proba_map.map_type == "Gaussian":
            return GaussianSPACS
        if proba_map.map_type == "PreExpFamily":
            return PreExpSPACS
        if proba_map.map_type == "ExponentialFamily":
            return SurPACSolver
        if pac_bayes_solver is None:
            return GradientBasedBayesSolver

        raise ValueError(
            f"'{pac_bayes_solver}' can only be used for Gaussian, Block diagonal Gaussian or Exponential Families (instances of 'PreExpFamily' or 'ExponentialFamily'"
        )
    elif isinstance(pac_bayes_solver, str):
        if pac_bayes_solver == "score_approx_gauss":
            if proba_map.map_type != "Gaussian":
                raise ValueError(
                    "'score_approx_gauss' can only be used for 'GaussianMap', 'BlockDiagGaussMap', 'FactCovGaussianMap', 'FixedCovGaussianMap' or 'TensorizedGaussianMap'"
                )
            return GaussianSPACS

        elif pac_bayes_solver == "score_approx_pre_exp":
            if proba_map.map_type != "PreExpFamily":
                raise ValueError(
                    "score_approx_pre_exp can only be used for PreExpFamily"
                )
            return PreExpSPACS

        elif pac_bayes_solver == "score_approx_exp":
            if proba_map.map_type != "ExponentialFamily":
                raise ValueError(
                    "score_approx_exp can only be used for ExponentialFamily"
                )
            return SurPACSolver

        elif pac_bayes_solver == "corr_weights":
            return GradientBasedBayesSolver
        elif pac_bayes_solver == "knn":
            return KNNBayesSolver

        else:
            raise ValueError(
                f"'pac_bayes_solver' must be one of {set_pac_bayes_solver} (value {pac_bayes_solver})"
            )

    else:
        return pac_bayes_solver


def pacbayes_minimize(
    fun: Callable[[np.ndarray], float],
    proba_map: ProbaMap,
    temperature: float,
    prior_param: Optional[ProbaParam] = None,
    optimizer: Optional[Union[str, type[BayesSolver]]] = None,
    parallel: bool = True,
    vectorized: bool = False,
    **kwargs,
) -> OptimResultBayes:
    """ Perform the minimization of Catoni's PAC-Bayes bound.
    Method used to perform the minimisation can be user specified or
    inferred from the type of proba_map.

    Args:
        fun: empirical risk function
        proba_map: ProbaMap object, space on which the minimization is performed
        temperature: PAC-Bayes temperature (the higher, the closer the posterior will be to the prior)
        optimizer: Optimizer class used to minimize Catoni's bound.
            "score_approx", "score_approx_gauss", "score_approx_pre_exp", "corr_weights", "knn" are
            also valid argument names and recognized. Default is None (inferred from proba_map).
        parallel: whether fun calls should be parallelized or not. Default is True.
        vectorized: whether fun is vectorized. Default is False. If True, parallelisation is deactivated.
    kwargs are passed to the optimizer (and uncaught kwargs passed to fun).
    """

    Optim = infer_pb_routine(proba_map=proba_map, pac_bayes_solver=optimizer)

    optim = Optim(
        fun=fun,
        proba_map=proba_map,
        prior_param=prior_param,
        temperature=temperature,
        parallel=parallel,
        vectorized=vectorized,
        **kwargs,
    )
    optim.optimize()

    return optim.process_result()
