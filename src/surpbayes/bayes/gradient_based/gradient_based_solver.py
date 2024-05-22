"""Gradient based solver
Contains both a vanilla GD implementation and more sophisticated
implementation with Momentum + reusing previous risk evaluations.
"""

import warnings
from typing import Callable, Optional, Union

import numpy as np
from scipy.stats import norm
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.bayes.bayes_solver import BayesSolver
from surpbayes.bayes.gradient_based.accu_sample_dens import AccuSampleValDens
from surpbayes.bayes.gradient_based.optim_result_vi_gb import OptimResultVIGB
from surpbayes.bayes.hist_vi import HistVILog
from surpbayes.misc import blab, par_eval, prod
from surpbayes.proba import Proba, ProbaMap
from surpbayes.types import ProbaParam, SamplePoint, Samples


class ProbBadGrad(Warning):
    """
    Warning class to indicate that a badly estimated gradient step has occured.
    """


def gen_eval_samp(
    proba: Proba,
    fun: Union[Callable[[SamplePoint], float], Callable[[Samples], np.ndarray]],
    n: int,
    parallel: bool = True,
    vectorized: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    sample = proba(n)
    l_dens = proba.log_dens(sample)
    if vectorized:
        vals = fun(sample, **kwargs)
    else:
        vals = par_eval(fun=fun, xs=sample, parallel=parallel, **kwargs)  # type: ignore
    return (sample, l_dens, vals)


class GDBayesSolver(BayesSolver):
    """
    Main class for Bayesian optimisation using the Gradient Descent method

    This class works for all types of ProbaMap and is used as the default implementation
    for all non exponential family ProbaMap.
    """
    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        proba_map: ProbaMap,
        prior_param: Optional[ProbaParam] = None,
        post_param: Optional[ProbaParam] = None,
        temperature: float = 1.0,
        prev_eval: Optional[AccuSampleVal] = None,
        index_train: Optional[list[int]] = None,
        eta: float = 0.05,
        chain_length: int = 10,
        per_step: Union[int, list[int]] = 100,
        kltol: float = 10**-8,
        xtol: float = 10**-8,
        n_grad_kl: int = 10**4,
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
            chain_length=chain_length,
            per_step=per_step,
            kltol=kltol,
            parallel=parallel,
            vectorized=vectorized,
            print_rec=print_rec,
            silent=silent,
            **kwargs,
        )

        self.eta = eta
        self.xtol = xtol

        self.n_grad_kl = n_grad_kl

        n_param = prod(self.prior_param.shape)
        if index_train is None:
            self.index_no_train = []
            self.index_train = list(range(n_param))
        else:
            self.index_no_train = list(set(range(n_param)).difference(index_train))
            self.index_train = list(index_train)

        self.grad_kl = self.proba_map.grad_kl(self.prior_param)

    def msg_begin_calib(self) -> None:
        blab(
            self.silent,
            " ".join(
                [
                    "Starting Bayesian calibration",
                    "(Gradient descent routine)",
                ]
            ),
        )

    def update(self):
        self.msg_begin_step()

        # Generate and evaluate new samples
        n_step = self.per_step[self.count]
        samples = self.post(n_step)
        if self.vectorized:
            vals = self.loc_fun(samples)
        else:
            vals = par_eval(fun=self.loc_fun, xs=samples, parallel=self.parallel)  # type: ignore

        self.accu.add(samples, vals)

        mval = np.mean(vals)

        der_log = self.proba_map.log_dens_der(self.post_param)
        grads = der_log(samples)
        grad_expct = np.tensordot((vals - mval), grads, (0,0)) / n_step


        grad_kl, kl = self.grad_kl(self._post_param, self.n_grad_kl)

        step = - self.eta * (grad_expct + self.temperature * grad_kl)
        score_VI = mval + self.temperature * kl
        self.hist_log.add1(proba_par=self._post_param, score=score_VI, KL=kl, mean=mval)

        new_post_param = self._post_param + step
        self.check_convergence(new_post_param)
        self.post_param = new_post_param
        
        self.count += 1
        self.msg_end_step()

    def msg_end_step(self):
        try:
            super().msg_end_step()
        except IndexError:
            print(f"No score assessed at step {self.count}")

    def optimize(self) -> None:
        self.msg_begin_calib()
        return super().optimize()



class GradientBasedBayesSolver(GDBayesSolver):
    """
    Bayesian optimisation using the Gradient Descent method with extra options

    This class works for all types of ProbaMap and is used as the default implementation
    for all non exponential family ProbaMap.

    Differences with GDBayesSolver:
        - Momentum can be used
        - Previous score evaluations are used through density corrected weight
    """

    accu_type = AccuSampleValDens

    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        proba_map: ProbaMap,
        prior_param: Optional[ProbaParam] = None,
        post_param: Optional[ProbaParam] = None,
        temperature: float = 1.0,
        prev_eval: Optional[AccuSampleVal] = None,
        index_train: Optional[list[int]] = None,
        eta: float = 0.05,
        chain_length: int = 10,
        per_step: Union[int, list[int]] = 100,
        kltol: float = 10**-8,
        xtol: float = 10**-8,
        k: Optional[int] = None,
        gen_decay: float = 0.0,
        momentum: float = 0.0,
        refuse_conf: float = 0.99,
        corr_eta: float = 0.5,
        n_grad_kl: int = 10**4,
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
            index_train=index_train,
            eta=eta * (1- momentum),
            chain_length=chain_length,
            per_step=per_step,
            kltol=kltol,
            xtol=xtol,
            n_grad_kl=n_grad_kl,
            parallel=parallel,
            vectorized=vectorized,
            print_rec=print_rec,
            silent=silent,
            **kwargs,
        )

        self.momentum = momentum

        self.refuse_factor = norm.ppf(refuse_conf)

        self.v = np.zeros(self.proba_map.proba_param_shape)

        self.prev_score = np.inf
        self.bin_log = HistVILog(proba_map, n=chain_length)
        self.all_log_vi = HistVILog(proba_map, n=chain_length)
        
        self.k = k
        self.corr_eta = corr_eta
        self.gen_decay = gen_decay

        # Back up if too many ProbaBadGrad
        self.ini_post_param = self._post_param.copy()


    def msg_begin_calib(self) -> None:
        blab(
            self.silent,
            " ".join(
                [
                    "Starting Bayesian calibration",
                    "(Gradient descent routine",
                    "with weight correction)",
                ]
            ),
        )

    def set_up_accu(self, prev_eval: Optional[AccuSampleVal]) -> None:
        if prev_eval is None:
            self.accu = AccuSampleValDens(
                sample_shape=self.proba_map.sample_shape, n_tot=sum(self.per_step)
            )
        else:
            self.accu = prev_eval  # type: ignore

            self.accu.extend_memory(sum(self.per_step))

        self.stable_accu = AccuSampleVal(sample_shape=self.proba_map.sample_shape, n_tot=sum(self.per_step))

    def gen_sample(self) -> None:

        sample, l_dens, vals = gen_eval_samp(
            proba=self._post,
            fun=self.fun,
            n=self.per_step[self.count],
            parallel=self.parallel,
            vectorized=self.vectorized,
            **self.kwargs,
        )

        self.accu.add(sample, l_dens, vals)
        self.stable_accu.add(sample, vals)

    def check_bad_grad(self, score_VI, score_UQ):
        """Checks if the score update lead to a bad"""
        is_bad = score_VI - self.prev_score > self.refuse_factor * score_UQ
        if is_bad:
            warnings.warn(
                f"""
            Harmful step removed.
            (Previous score: {self.prev_score}, new_score: {score_VI}, UQ: {score_UQ}))""",
                category=ProbBadGrad,
            )
        return is_bad

    def get_score_grad(self) -> tuple[np.ndarray, float, float]:
        """Compute gradient of score wrt to ProbaParam

        Returns a tuple containing the gradient, the mean of the score
        and the uncertainty on the mean of the score
        """
        return self.accu.grad_score(
            self.proba_map, self._post_param, k=self.k, gen_decay=self.gen_decay
        )

    def mod_accu_bad_step(self):
        """Removed from main function to ease modification for KNN"""
        self.accu.suppr_gen(2)

    def update(self):
        self.msg_begin_step()

        # Generate and evaluate new samples
        self.gen_sample()

        # Move parameter accordingly
        der_score, m_score, score_UQ = self.get_score_grad()
        der_KL, kl = self.grad_kl(self._post_param, self.n_grad_kl)

        score_VI = m_score + self.temperature * kl

        if self.check_bad_grad(score_VI, score_UQ):

            self.v = np.zeros(self.proba_map.proba_param_shape)
            self.eta = self.eta * self.corr_eta

            # Go back one generation and store deleted information
            self.mod_accu_bad_step()
            self.bin_log.add(*self.hist_log.suppr(1))  # type: ignore

            try:
                self._post_param = self.hist_log.proba_pars()[-1]
                self._post = self.proba_map(self._post_param)  # type: ignore
                self.prev_score = self.hist_log.VI_scores()[-1]
            except IndexError:
                # If the above fails, means back to the beginning (no previous data logged)
                # Set previous score to infinity
                self._post_param = self.ini_post_param.copy()
                self._post = self.proba_map(self._post_param)
                self.prev_score = np.inf

        else:
            self.hist_log.add1(proba_par=self._post_param, score=score_VI, KL=kl, mean=m_score)  # type: ignore

            v_new = -self.eta * (der_score + self.temperature * der_KL)
            v_new_flat = v_new.flatten()
            v_new_flat[self.index_no_train] = 0.0
            v_new = v_new_flat.reshape(self.proba_map.proba_param_shape)
            self.v = v_new + self.v * self.momentum

            new_post_param = self._post_param + self.v
            self.check_convergence(new_post_param)

            self._post_param = new_post_param
            self.prev_score = score_VI
            self._post = self.proba_map(self._post_param)

        self.count += 1
        self.msg_end_step()

    def msg_end_step(self):
        try:
            super().msg_end_step()
        except IndexError:
            print(f"No score assessed at step {self.count}")

    def process_result(self) -> OptimResultVIGB:
        return OptimResultVIGB(
            opti_param=self._post_param,
            converged=self.converged,
            opti_score=self.hist_log.VI_scores(1)[0],
            hist_param=self.hist_log.proba_pars(),
            hist_score=self.hist_log.VI_scores(),
            end_param=self._post_param,
            log_vi=self.hist_log,
            bin_log_vi=self.bin_log,
            sample_val=self.accu,
        )
