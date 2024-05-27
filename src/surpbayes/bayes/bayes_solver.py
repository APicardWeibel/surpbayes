from functools import partial
from typing import Callable, Optional, Union

import numpy as np
from multiprocess import Pool
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.bayes.hist_bayes import HistBayesLog
from surpbayes.bayes.optim_result_bayes import OptimResultBayes
from surpbayes.misc import blab, prod
from surpbayes.proba import ProbaMap
from surpbayes.types import ProbaParam


class BayesSolver:
    r"""
    Main class for PAC-Bayes objective minimisation.

    For a map to probability distributions \pi, BayesSolver solves the following
    minimisation task:

    argmin_{\theta} J(\theta) = \pi(\theta)[fun] + temperature * KL(\pi(\theta), \pi(\theta_0))

    NOTE:
        At this stage, BayesSolver class is not inherited from GenOptimizer
    """

    accu_type = AccuSampleVal

    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        proba_map: ProbaMap,
        prior_param: Optional[ProbaParam] = None,
        post_param: Optional[ProbaParam] = None,
        temperature: float = 1.0,
        prev_eval: Optional[AccuSampleVal] = None,
        chain_length: int = 10,
        per_step: Union[int, list[int]] = 100,
        xtol: float = 10**-8,
        kltol: float = 10**-8,
        # For function evaluation
        parallel: bool = True,
        vectorized: bool = False,
        print_rec: int = 1,
        silent: bool = False,
        **kwargs,
    ):

        self.proba_map = proba_map
        self.fun = fun
        self.proba_map = proba_map

        # Function evaluation rules
        if vectorized:
            self.parallel = False
        else:
            self.parallel = parallel
        self.vectorized = vectorized

        if self.parallel:
            self.pool = Pool()  # pylint: disable=E1102
        else:
            self.pool = None

        if prior_param is None:
            self.prior_param = self.proba_map.ref_param
        else:
            # Force convert to array
            prior_param_arr = np.asarray(prior_param)
            assert prior_param_arr.shape == self.proba_map.proba_param_shape
            self.prior_param = prior_param_arr.copy()

        if post_param is None:
            self._post_param = self.prior_param
        else:
            # Force convert to array
            post_param_arr = np.asarray(post_param)
            assert post_param_arr.shape == self.proba_map.proba_param_shape
            self._post_param = post_param_arr.copy()

        self._post = proba_map(self._post_param)

        self.temperature = temperature
        self.chain_length = chain_length

        self.silent = silent
        self.print_rec = print_rec

        self.per_step = per_step

        self.xtol = xtol
        self.kltol = kltol

        self.kwargs = kwargs
        self.loc_fun = partial(self.fun, **kwargs)
        self.set_up_per_step(per_step)
        self.set_up_accu(prev_eval)

        self.hist_log = HistBayesLog(proba_map, chain_length + 1)

        self.converged = False
        self.count = 0

    @property
    def post(self):
        """Posterior distribution"""
        return self._post

    @property
    def post_param(self):
        return self._post_param

    @post_param.setter
    def post_param(self, value):
        value_arr = np.asarray(value)
        if value_arr.shape != self.proba_map.proba_param_shape:
            raise ValueError(
                "post_param shape passed is not compatible with expected proba_map"
            )
        self._post_param = value_arr
        self._post = self.proba_map(value_arr)

    def set_up_per_step(self, per_step: Optional[Union[int, list[int]]]) -> None:
        proba_dim = prod(self.proba_map.proba_param_shape)
        if per_step is None:
            self.per_step = [max(100, 2 * proba_dim) for _ in range(self.chain_length)]
        elif isinstance(per_step, int):
            self.per_step = [per_step for _ in range(self.chain_length)]
        else:
            self.per_step = per_step

    def set_up_accu(self, prev_eval: Optional[AccuSampleVal]) -> None:
        """Prepare accu from optional previous accu
        Arg:
            prev_eval, an optional AccuSampleVal instance.
        """
        if prev_eval is None:
            self.accu = AccuSampleVal(
                sample_shape=self.proba_map.sample_shape, n_tot=sum(self.per_step)
            )
        else:
            self.accu = prev_eval
            self.accu.extend_memory(sum(self.per_step))

    def update(self):
        """Dummy update method - must be reimplemented!"""
        self.msg_begin_step()

        self.hist_log.add1(
            proba_par=self._post_param, score=np.nan, KL=np.nan, mean=np.nan
        )

        self.count += 1
        self.msg_end_step()

    def optimize(self) -> None:
        """Main Optimisation call
        Loops update until convergence or max chain length exceeded"""
        try:
            while (self.count < self.chain_length) & (not self.converged):
                self.update()
            if self.parallel:
                self.pool.close()
        except Exception as exc:
            # Terminate pool of worker if parallel
            if self.parallel:
                self.pool.terminate()
            raise exc

    def process_result(self) -> OptimResultBayes:
        return OptimResultBayes(
            opti_param=self._post_param,
            converged=self.converged,
            opti_score=self.hist_log.bayes_scores(1)[0],
            hist_param=self.hist_log.proba_pars(),
            hist_score=self.hist_log.bayes_scores(),
            end_param=self._post_param,
            log_bayes=self.hist_log,
            sample_val=self.accu,
        )

    def msg_begin_calib(self) -> None:
        blab(self.silent, "Starting PAC-Bayes training")

    def msg_begin_step(self) -> None:
        """Print at the beginning of a step"""

    def msg_end_step(self) -> None:
        """Print at the end of a step"""
        silent_loc = self.silent or (self.count % self.print_rec != 0)
        _, tot_score, kl, mean_score = self.hist_log.get(1)
        blab(
            silent_loc,
            f"Score at step {self.count}/{self.chain_length}: {tot_score[0]} (KL: {kl[0]}, score:{mean_score[0]})",
        )

    def msg_end_calib(self) -> None:
        _, tot_score, kl, mean_score = self.hist_log.get(1)
        msg = "End of PAC-Bayes training."
        if self.converged:
            msg = msg + " Converged.\n"
        else:
            msg = msg + " Did not converge.\n"

        msg += f"End score:  {tot_score[0]} (KL: {kl[0]}, score:{mean_score[0]})"
        blab(
            self.silent,
            msg,
        )

    def check_convergence(self, new_post_param):
        """Default convergence check"""
        self.converged = (
            np.all(np.abs(self._post_param - new_post_param) < self.xtol)
            or self.proba_map.kl(new_post_param, self._post_param) < self.kltol
        )
