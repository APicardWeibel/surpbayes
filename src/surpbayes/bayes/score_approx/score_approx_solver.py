from typing import Callable, Optional, Union

import numpy as np
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.bayes.bayes_solver import BayesSolver
from surpbayes.bayes.optim_result_vi import OptimResultVI
# from surpbayes.bayes.score_approx._helper import set_up_per_step
from surpbayes.bayes.score_approx.accu_sample_exp import (AccuSampleValExp,
                                                     _add_T_data)
from surpbayes.bayes.score_approx.weighing import get_weights_mc
from surpbayes.misc import blab, prod
from surpbayes.optim import dichoto
from surpbayes.proba import ExponentialFamily, PreExpFamily, Proba, RenormError
from surpbayes.types import ProbaParam, SamplePoint, Samples

# from multiprocess import Pool  # pylint: disable=E0611


def _set_up_per_step(
    per_step: Optional[Union[int, list[int]]],
    chain_length: int,
    n_max_eval: Optional[int],
    proba_dim: int,
    silent,
) -> list[int]:
    if per_step is None:
        per_step = [max(100, 2 * proba_dim)]
        if chain_length > 1:
            per_step = per_step + [
                max(50, proba_dim // 2) for _ in range(chain_length - 1)
            ]

    if isinstance(per_step, int):
        if n_max_eval is None:
            per_step = [per_step] * chain_length
        elif (per_step * chain_length) <= n_max_eval:
            per_step = [per_step] * chain_length
        else:
            n_full = n_max_eval // per_step
            _remain = n_max_eval % per_step
            per_step = [per_step] * n_full + [_remain]
            if n_full + 1 < chain_length:
                per_step = per_step + [0] * (chain_length - n_full - 1)

    else:
        # Limit per_step to first chain_length arguments
        per_step = per_step[:chain_length]

        # Check if necessary to modify per_step
        cum_steps = np.cumsum(per_step)

        if (n_max_eval is not None) and (cum_steps[-1] > n_max_eval):
            blab(
                silent,
                "Warning: per_step indication exceeds number of allowed evaluations",
            )
            cut_off = np.sum(cum_steps < n_max_eval)
            if cut_off > 0:
                per_step[cut_off] = n_max_eval - cum_steps[cut_off - 1]
            else:
                per_step[cut_off] = n_max_eval
            for i in range(cut_off + 1, len(per_step)):
                per_step[i] = 0

    return per_step


def exp_approximation(
    Ts: Samples,
    scores: np.ndarray,
    weights: Optional[np.ndarray] = None,
):
    r"""Compute the best approximation of the score function as $x \rightarrow \theta\cdot T(x)$
    using a weighted L2 criteria from T(x_i), S(x_i) and weights

    """
    if weights is None:
        weights = np.ones(len(scores)) / len(scores)

    # Check Ts implementation
    # Ts = proba_map.T(xs)

    Ts_center = Ts - np.tensordot(weights, Ts, (0, 0))

    solver = np.linalg.inv(np.tensordot(Ts_center, weights * Ts_center.T, (0, 1)))

    # Compute scalar product with scores
    scalar_prod = np.tensordot(scores * weights, Ts_center, (0, 0))
    return solver @ scalar_prod


def _solve_in_kl(
    proba_map: ExponentialFamily,
    post_param: ProbaParam,
    direction: np.ndarray,
    kl_max: float,
    alpha_max: float,
    y_pres: Optional[float] = None,
    x_pres: Optional[float] = None,
    m_max=100,
):
    r"""Compute the highest alpha such that
    Kl(\pi(post_param + \alpha post_param), \pi(post_param))< kl_max
    for alpha < alpha_max

    Implemented for proba_map being an exponential family.
    Solved using dichotomy
    """

    ### Aims at solving proba_map.kl( proba_map.to_param(), post_param)
    def loc_fun(alpha):
        try:
            out = proba_map.kl(post_param + alpha * direction, post_param)

        except RenormError:
            out = np.inf
        return out

    # Use dichotomy, take lower value
    if loc_fun(alpha_max) < kl_max:
        return alpha_max

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


class ScoreApproxBayesSolver(BayesSolver):
    """
    Main class for Bayesian optimisation using the score approximation method

    ------------------------------- BACKGROUND -------------------------------

    ScoreApproxBayesSolver strives to minimize Catoni's bound through an score
    approximation method on exponential families

    For a parametric family of distribution 'proba_map' (noted $\pi(\theta)$),
    Catoni's bound is defined as

        $$\pi(\theta)[S] + temperature * KL(\pi(\theta), \pi_0) $$

    For exponential families, the prior to posterior transform amounts to
    learning the best approximation of the score of form $\theta \cdot T(x)$
    for the L2 criteria at the posterior.

    The learning algorithm is therefore performed by alternating learning score
    approximation from posterior approximation and learning posterior
    approximation from score approximation.

    To improve robustness of the algorithm, the new posterior approximation is
    allowed to be at most at a distance of 'kl_max' of the current posterior
    approximation.

    The integration scheme used to approximate the score uses all previously
    used samples. Weights are computed by approximating the weights given by
    the posterior approximation to a partition of the space defined by the
    drawned samples (each set containing exactly one sample).

    ----------------------------- IMPLEMENTATION -----------------------------

    This class works for proba_map an instance of ExponentialFamily, and uses
    all the default routines for the sub tasks. Custom implementation of the
    sub routines can be used by defining a subclass and reimplementing the
    adequate method. The main methods, "update" and "optimize" should not be
    reimplemented.

    Main steps:
        gen_sample
        lin_reg
        get_t_dir
        get_updt_speed
        updt_post_par

    -------------------------------- ARGUMENTS -------------------------------
    Args:
        fun: a scoring function
        proba_map: parametric set of distributions
        prior_param: parameter describing the prior distribution. Optional (if
            None, uses proba_map.ref_proba_param)
        post_param: parameter describing the initial guess of posterior
            distribution. Optional (if None, uses prior_param)
        temperature: the PAC-Bayesian temperature used to construct the
            posterior distribution
        prev_eval: AccuSampleVal object encoding previous evaluations of the
            function (useful for retraining). Optional.
        n_estim_weights: numbers of samples from current distribution drawned
            to estimate the weight of each score evaluated samples. Default is
            10**5.
        kl_max: float (positive), maximum step size (in kl) between current and
            new posterior (in the sense that
            kl(post_{n+1}, post_{n}) <= kl_max). Default is 10.0
        dampen: float (0<= < 1). Contol maximum update speed of posterior. From
            score approximation $\theta$, the posterior distribution should be
            $\theta_0-temperature^{-1} \theta$. Using dampen >0, the update is
                changed to
                $\theta_{n+1}
                =\theta_n
                + (1-dampen) (\theta_0-temperature^{-1} \theta - \theta_n)$
            The fixed point solution of the problem remains unchanged, but
            prevents relying to much on local approximations of the score.
        chain_length: number of posterior updates.
        n_max_eval: maximum number of calls to fun (counting previous calls
            inferred from prev_eval)
        per_step: number of draws from the prior evaluated through 'fun' at
            update step
        kltol: termination criteria (stops when
            kl(post_{n+1}, post_{n}) <= kltol)
        y_pres: argument for dichotomy solving when enforcing
            kl(post_{n+1}, post_{n}) <= kl_max
        x_pres: argument for dichotomy solving when enforcing
            kl(post_{n+1}, post_{n}) <= kl_max
        m_max: argument for dichotomy solving when enforcin
             kl(post_{n+1}, post_{n}) <= kl_max
        parallel: should 'fun' evaluations be parallelized? Default is True
        vectorized: is 'fun' vectorized? If True, 'parallel' is disregarded
        print_rec: How many gradient steps should there be prints
        silent: should there be any prints at all ? Default is False
            (there are prints)
    Further **kwargs will be passed to "fun"
    """

    accu_type = AccuSampleValExp

    def __init__(
        self,
        fun: Callable[[SamplePoint], float],
        proba_map: Union[PreExpFamily, ExponentialFamily],
        prior_param: Optional[ProbaParam] = None,
        post_param: Optional[ProbaParam] = None,
        temperature: float = 1.0,
        prev_eval: Optional[AccuSampleVal] = None,
        # for weighing
        n_estim_weights: int = 10**5,
        k_neighbors: int = 1,
        # For smoothing training path
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
        """
        Args:
            fun: a scoring function
            proba_map: parametric set of distributions
            prior_param: parameter describing the prior distribution. Optional (if
                None, uses proba_map.ref_proba_param)
            post_param: parameter describing the initial guess of posterior
                distribution. Optional (if None, uses prior_param)
            temperature: the PAC-Bayesian temperature used to construct the
                posterior distribution
            prev_eval: AccuSampleVal object encoding previous evaluations of the
                function (useful for retraining). Optional.
            n_estim_weights: numbers of samples from current distribution drawned
                to estimate the weight of each score evaluated samples. Default is
                10**5.
            kl_max: float (positive), maximum step size (in kl) between current and
                new posterior (in the sense that
                kl(post_{n+1}, post_{n}) <= kl_max). Default is 10.0
            dampen: float (0<= < 1). Contol maximum update speed of posterior. From
                score approximation $\theta$, the posterior distribution should be
                $\theta_0-temperature^{-1} \theta$. Using dampen >0, the update is
                    changed to
                    $\theta_{n+1}
                    =\theta_n
                    + (1-dampen) (\theta_0-temperature^{-1} \theta - \theta_n)$
                The fixed point solution of the problem remains unchanged, but
                prevents relying to much on local approximations of the score.
            chain_length: number of posterior updates.
            n_max_eval: maximum number of calls to fun (counting previous calls
                inferred from prev_eval)
            per_step: number of draws from the prior evaluated through 'fun' at
                update step
            kltol: termination criteria (stops when
                kl(post_{n+1}, post_{n}) <= kltol)
            y_pres: argument for dichotomy solving when enforcing
                kl(post_{n+1}, post_{n}) <= kl_max
            x_pres: argument for dichotomy solving when enforcing
                kl(post_{n+1}, post_{n}) <= kl_max
            m_max: argument for dichotomy solving when enforcin
                 kl(post_{n+1}, post_{n}) <= kl_max
            parallel: should 'fun' evaluations be parallelized? Default is True
            vectorized: is 'fun' vectorized? If True, 'parallel' is disregarded
            print_rec: How many gradient steps should there be prints
            silent: should there be any prints at all ? Default is False
                (there are prints)
        Further **kwargs will be passed to "fun"
        """
        # n_max_eval must be an attribute for set_up_per_step to work
        # when called by super().__init__
        if (n_max_eval is not None) & (prev_eval is not None):
            self.n_max_eval = n_max_eval - prev_eval.n_filled
        else:
            self.n_max_eval = n_max_eval

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
        self.inv_temp = 1 / self.temperature

        self.n_estim_weights = n_estim_weights
        self.k_neighbors = k_neighbors

        self.kl_max = kl_max
        self.dampen = dampen

        if y_pres is None:
            y_pres = 0.02 * kl_max

        self.y_pres = y_pres
        self.x_pres = x_pres
        self.m_max = m_max

        self._post = self.proba_map(self._post_param)

    def msg_begin_calib(self) -> None:
        """Message printed at the beginning of the calibration routine"""
        blab(
            self.silent,
            " ".join(
                [
                    "Starting Bayesian calibration",
                    "(Score Approximation routine)",
                ]
            ),
        )

    def set_up_per_step(self, per_step: Optional[Union[int, list[int]]]) -> None:
        """
        Per step set up for BayesSolver.
        The resulting per_step argument stored is a list of int of size chain_length
        """
        self.per_step = _set_up_per_step(
            per_step,
            chain_length=self.chain_length,
            n_max_eval=self.n_max_eval,
            proba_dim=prod(self.proba_map.proba_param_shape),
            silent=self.silent,
        )
        self.tot_draw = sum(self.per_step)

    def set_up_accu(self, prev_eval: Optional[AccuSampleVal]) -> None:
        if prev_eval is None:
            self.accu = AccuSampleValExp(
                sample_shape=self.proba_map.sample_shape,
                t_shape=self.proba_map.t_shape,
                n_tot=self.tot_draw,
            )
        else:

            self.accu = prev_eval  # type: ignore
            if not isinstance(self.accu, AccuSampleValExp):
                # Breaks dependance
                self.accu = _add_T_data(self.accu, proba_map=self.proba_map)

            n_remain = self.accu.n_remain()
            if self.tot_draw > n_remain:
                self.accu.extend_memory(self.tot_draw - n_remain)

    def weigh(self, proba: Proba, samples: Samples, n_sample_estim: int) -> np.ndarray:
        """
        Default weighing process for score_approx routine. The weight given to each sample
        is an approximation of the probability mass given by proba to the Voronoi cell (for
        the standard euclidean distance). This mass is estimated by nearest neighbor searches
        on 'n_sample_estim' fresh samples generated from "proba" (performed using faiss library)
        """
        return get_weights_mc(
            proba, samples, n_sample_estim=n_sample_estim, k_neighbors=self.k_neighbors
        )

    def add_hist_log(self, weights: np.ndarray) -> None:
        """Log performance"""
        kl = self.proba_map.kl(self._post_param, self.prior_param)
        mean_score = np.sum(weights * self.accu.vals())
        tot_score = mean_score + self.temperature * kl
        self.hist_log.add1(self._post_param, score=tot_score, KL=kl, mean=mean_score)

    def gen_sample(self) -> None:
        """
        Generate and evaluate new samples.

        Default implementation:
        Samples are generated at random from the posterior distribution.

        Sample evaluations are added inplace to 'accu' attribute
        """
        try:
            new_samp = self._post(self.per_step[self.count])
        except Exception as exc:
            print(self._post_param)
            raise exc

        if self.parallel:
            new_Ss = np.array(self.pool.map(self.loc_fun, new_samp))

        elif self.vectorized:
            new_Ss = self.loc_fun(new_samp)

        else:
            new_Ss = np.array([self.loc_fun(x) for x in new_samp])

        self.accu.add(new_samp, new_Ss, self.proba_map.T(new_samp))

    def lin_reg(
        self, Ts: np.ndarray, scores: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Linear regression solver for score(x) = theta . T(x) + c
        weighted by weights. c is not returned
        """
        if Ts.shape[0] < Ts.shape[1]:
            raise ValueError("Linear regression failure: not enough samples")
        return exp_approximation(Ts=Ts, scores=scores, weights=weights)

    def get_t_dir(self, t_score: np.ndarray) -> np.ndarray:
        """Infer update direction for the natural
        parametrisation of the exponential family"""
        return self.prior_param - self.inv_temp * t_score - self._post_param

    def get_updt_speed(self, t_dir: np.ndarray) -> float:
        return _solve_in_kl(
            proba_map=self.proba_map,
            post_param=self._post_param,
            direction=t_dir,
            kl_max=self.kl_max,
            alpha_max=(1 - self.dampen),
            y_pres=self.y_pres,
            x_pres=self.x_pres,
            m_max=self.m_max,
        )

    def updt_post_par(self, t_dir: np.ndarray, alpha: float) -> None:
        new_post_param = self._post_param + alpha * t_dir
        self.check_convergence(new_post_param)
        self._post_param = new_post_param
        self._post = self.proba_map(self._post_param)

    def update(self) -> None:
        """Perform a single optimisation step"""
        self.msg_begin_step()
        self.gen_sample()

        weights = self.weigh(
            proba=self._post,
            samples=self.accu.sample(),
            n_sample_estim=self.n_estim_weights,
        )

        # Log to history
        self.add_hist_log(weights)

        # Compute best score approx
        t_score = self.lin_reg(
            Ts=self.accu.ts(), scores=self.accu.vals(), weights=weights
        )

        # Update posterior
        t_dir = self.get_t_dir(t_score)
        alpha = self.get_updt_speed(t_dir)
        self.updt_post_par(t_dir, alpha)

        self.count += 1
        self.msg_end_step()

    def eval_score(self) -> None:
        weights = self.weigh(
            proba=self._post,
            samples=self.accu.sample(),
            n_sample_estim=self.n_estim_weights,
        )

        # Log to history
        self.add_hist_log(weights)

    def optimize(self) -> None:
        """Optimisation call. Loops update until convergence or exceeds max chain length"""
        self.msg_begin_calib()

        # Main loop
        super().optimize()

        # Compute last score
        self.eval_score()

        # Print last message
        self.msg_end_calib()

        if self.parallel:
            self.pool.close()

    def process_result(self) -> OptimResultVI:
        """Return solver result as an OptimResultVI"""
        return OptimResultVI(
            opti_param=self._post_param,
            converged=self.converged,
            opti_score=self.hist_log.VI_scores(1)[0],
            hist_param=self.hist_log.proba_pars(),
            hist_score=self.hist_log.VI_scores(),
            end_param=self._post_param,
            log_vi=self.hist_log,
            sample_val=self.accu,
        )
