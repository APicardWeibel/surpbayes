from typing import Optional, Union

from surpbayes.bayes import BayesSolver, pacbayes_minimize
from surpbayes.misc import par_eval
from surpbayes.pyadm1.digester import Digester
from surpbayes.pyadm1.proba.standard_proba import convert_param
from surpbayes.pyadm1.proba.standard_proba import prior_param as default_prior_param
from surpbayes.pyadm1.proba.standard_proba import proba_map
from surpbayes.types import ProbaParam, SamplePoint, Samples


class Interface:
    """Interface helper to perform bayesian calibration in context with ADM1"""

    def __init__(
        self,
        digester: Digester,
        temperature: float = 1.0,
        prior_param: Optional[ProbaParam] = None,
    ):
        self.dig = digester
        self.temperature = temperature
        if prior_param is None:
            prior_param = default_prior_param
        self.prior_param = prior_param

    def score(
        self,
        par: SamplePoint,
        solver_method: str = "LSODA",
        max_step: float = 60.0 / (24.0 * 60.0),
        min_step: float = 10**-6,
        # Score arguments
        eps: float = 10**-8,
        max_score: float = 3.0,
        elbow: float = 2.0,
        **kwargs,
    ) -> float:
        return self.dig.score(
            param=convert_param(par),
            solver_method=solver_method,
            max_step=max_step,
            min_step=min_step,
            eps=eps,
            max_score=max_score,
            elbow=elbow,
            **kwargs,
        )

    def mult_score(
        self,
        pars: Samples,
        solver_method: str = "LSODA",
        max_step: float = 60.0 / (24.0 * 60.0),
        min_step: float = 10**-6,
        # Score arguments
        eps: float = 10**-8,
        max_score: float = 3.0,
        elbow: float = 2.0,
        **kwargs,
    ):
        """Multiple calls to score, parallelized"""
        return par_eval(
            self.score,
            pars,
            parallel=True,
            solver_method=solver_method,
            max_step=max_step,
            min_step=min_step,
            eps=eps,
            max_score=max_score,
            elbow=elbow,
            **kwargs,
        )

    def bayes_calibration(
        self,
        optimizer: Optional[Union[str, type[BayesSolver]]] = None,
        solver_method: str = "LSODA",
        max_step: float = 60.0 / (24.0 * 60.0),
        min_step: float = 10**-6,
        # Score arguments
        eps: float = 10**-8,
        max_score: float = 3.0,
        elbow: float = 2.0,
        **kwargs,
    ):
        return pacbayes_minimize(
            self.score,
            proba_map=proba_map,
            temperature=self.temperature,
            prior_param=self.prior_param,
            optimizer=optimizer,
            parallel=True,
            vectorized=False,
            solver_method=solver_method,
            max_step=max_step,
            min_step=min_step,
            # Score arguments
            eps=eps,
            max_score=max_score,
            elbow=elbow,
            **kwargs,
        )
