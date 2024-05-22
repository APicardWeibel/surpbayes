from typing import Callable, Optional, Type, Union

import numpy as np

from surpbayes.bayes.bayes_solver import BayesSolver
from surpbayes.bayes.gradient_based import (
    GradientBasedBayesSolver,
    KNNBayesSolver,
)
from surpbayes.bayes.optim_result_vi import OptimResultVI
from surpbayes.bayes.score_approx import (
    GaussianSABS,
    PreExpSABS,
    ScoreApproxBayesSolver,
)
from surpbayes.proba import ProbaMap

set_VI_method = {
    "corr_weights",
    "knn",
    "score_approx",
    "score_approx_gauss",
    "score_approx_pre_exp",
    "score_approx_exp",
}


def infer_VI_routine(
    proba_map: ProbaMap, VI_method: Optional[Union[str, Type[BayesSolver]]] = None
) -> Type[BayesSolver]:
    """Infer which VI_method from 'proba_map' and 'VI_method' arguments.

    Check Coherence between 'VI_method' and 'proba_map'.

    Rules:
        If None, defaults to 'corr_weights' for generic distribution and the adequate
        'score_approx' routines for ExponentialFamily, PreExpFamily, and Gaussian related
        distributions.
        If 'score_approx', checks the appropriate version of 'score_approx' depending on the
        'proba_map' passed.
    """
    if (VI_method is None) or (VI_method == "score_approx"):
        if proba_map.map_type == "Gaussian":
            return GaussianSABS
        if proba_map.map_type == "PreExpFamily":
            return PreExpSABS
        if proba_map.map_type == "ExponentialFamily":
            return ScoreApproxBayesSolver
        if VI_method is None:
            return GradientBasedBayesSolver

        raise ValueError(
            "'score_approx' can only be used for Gaussian, Block diagonal Gaussian or Exponential Families"
        )
    elif isinstance(VI_method, str):
        if VI_method == "score_approx_gauss":
            if proba_map.map_type != "Gaussian":
                raise ValueError(
                    "'score_approx_gauss' can only be used for 'GaussianMap', 'BlockDiagGaussMap', 'FactCovGaussianMap', 'FixedCovGaussianMap' or 'TensorizedGaussianMap'"
                )
            return GaussianSABS

        elif VI_method == "score_approx_pre_exp":
            if proba_map.map_type != "PreExpFamily":
                raise ValueError(
                    "score_approx_pre_exp can only be used for PreExpFamily"
                )
            return PreExpSABS

        elif VI_method == "score_approx_exp":
            if proba_map.map_type != "ExponentialFamily":
                raise ValueError(
                    "score_approx_exp can only be used for ExponentialFamily"
                )
            return ScoreApproxBayesSolver

        elif VI_method == "corr_weights":
            return GradientBasedBayesSolver
        elif VI_method == "knn":
            return KNNBayesSolver

        else:
            raise ValueError(
                f"'VI_method' must be one of {set_VI_method} (value {VI_method})"
            )

    else:
        return VI_method


def variational_inference(
    fun: Callable[[np.ndarray], float],
    proba_map: ProbaMap,
    temperature: float,
    optimizer: Optional[Union[str, type[BayesSolver]]] = None,
    parallel: bool = True,
    vectorized: bool = False,
    **kwargs,
) -> OptimResultVI:

    Optim = infer_VI_routine(proba_map=proba_map, VI_method=optimizer)

    optim = Optim(
        fun=fun,
        proba_map=proba_map,
        temperature=temperature,
        parallel=parallel,
        vectorized=vectorized,
        **kwargs,
    )
    optim.optimize()

    return optim.process_result()
