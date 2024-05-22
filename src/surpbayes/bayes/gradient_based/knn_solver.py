from typing import Callable, Optional, Union

import numpy as np
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.bayes.gradient_based.gradient_based_solver import \
    GradientBasedBayesSolver
from surpbayes.misc import blab, par_eval
from surpbayes.proba import ProbaMap
from surpbayes.types import ProbaParam, SamplePoint, Samples


class KNNBayesSolver(GradientBasedBayesSolver):
    """Gradient based Bayesian using KNN built proxy for score

    No longer maintained

    Function calls recycling is achieved by building proxy function with KNN.

    If vectorized, it is assumed that the score function is vectorized (i.e one can directly
    evaluate multiple scores using fun(samples))
    """

    accu_type = AccuSampleVal

    def __init__(
        self,
        fun: Union[Callable[[SamplePoint], float], Callable[[Samples], np.ndarray]],
        proba_map: ProbaMap,
        prior_param: Optional[ProbaParam] = None,
        post_param: Optional[ProbaParam] = None,
        temperature: float = 1.0,
        prev_eval: Optional[AccuSampleVal] = None,
        index_train: Optional[list[int]] = None,
        eta: float = 0.05,
        chain_length: int = 10,
        per_step: int = 100,
        per_step_eval: int = 10000,
        kltol: float = 10**-8,
        xtol: float = 10**-8,
        k: Optional[int] = None,
        momentum: float = 0.0,
        # refuse_conf: float = 0.99,
        corr_eta: float = 0.5,
        # Compute gradient of KL in general case
        n_grad_kl: int = 10**4,
        parallel: bool = True,
        vectorized: bool = False,
        print_rec: int = 1,
        silent=False,
        **kwargs,
    ):
        super().__init__(
            fun=fun,
            proba_map=proba_map,
            prior_param=prior_param,
            post_param=post_param,
            temperature=temperature,
            prev_eval=prev_eval,
            index_train=index_train,
            eta=eta,
            chain_length=chain_length,
            per_step=per_step,
            kltol=kltol,
            xtol=xtol,
            k=k,
            momentum=momentum,
            corr_eta=corr_eta,
            n_grad_kl=n_grad_kl,
            parallel=parallel,
            vectorized=vectorized,
            print_rec=print_rec,
            silent=silent,
            **kwargs,
        )
        self.per_step_eval = per_step_eval

    def msg_begin_calib(self) -> None:
        blab(
            self.silent,
            " ".join(
                [
                    "Starting Bayesian calibration",
                    "(Gradient descent routine",
                    "with KNN trained proxy score)",
                ]
            ),
        )

    def set_up_accu(self, prev_eval: Optional[AccuSampleVal]) -> None:
        if prev_eval is None:
            self.accu = AccuSampleVal(self.proba_map.sample_shape, sum(self.per_step))
        else:
            self.accu = prev_eval
            self.accu.extend_memory(sum(self.per_step))

    def gen_sample(self):
        sample = self._post(self.per_step[self.count])
        if self.vectorized:
            vals = self.fun(sample, **self.kwargs)
        else:
            vals = par_eval(fun=self.fun, xs=sample, parallel=self.parallel, **self.kwargs)  # type: ignore
        # Store new sample
        self.accu.add(sample, vals)  # type: ignore

    def check_bad_grad(self, score_VI:float, UQ:float):
        """ Check if Bad gradient"""
        is_bad = score_VI > self.prev_score
        if is_bad:
            blab(
                self.silent,
                f"Undo last step (last score {self.prev_score}, new score {score_VI})",
            )
        return is_bad

    def get_score_grad(self):
        interpol = self.accu.knn(self.k)
        der_log = self.proba_map.log_dens_der(self._post_param)

        l_sample = self._post(self.per_step_eval)
        l_vals = interpol(l_sample)
        l_grads_log = der_log(l_sample)

        m_score = np.mean(l_vals)
        der_score = (
            np.tensordot((l_vals - m_score), l_grads_log, (0, 0)) / self.per_step_eval
        )

        return der_score, m_score, 0.0

    def mod_accu_bad_step(self):
        """For KNN setting, all evaluations are useful, so pass"""
        return None
