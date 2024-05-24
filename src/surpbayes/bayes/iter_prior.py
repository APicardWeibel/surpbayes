"""
Iter prior algorithm, inspired by A. Leurent and R. Moscoviz (https://doi.org/10.1002/bit.28156)

- OptimResultPriorIter subclass of OptimResult for iter_prior function
- iter_prior function
- iter_prior_bayes, adaptation of iter_prior for minimisation of PAC-Bayes bound over Gaussian.
"""

import os
import sys
from typing import Callable, Optional, Sequence, Union

import numpy as np
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.misc import blab, par_eval
from surpbayes.optim import OptimResult
from surpbayes.proba import TensorizedGaussianMap


class OptimResultPriorIter(OptimResult):
    """Subclass of OptimResult for prior_iter algorithm"""

    class_name = "OptimResultPriorIter"

    def __init__(
        self,
        opti_param,
        converged: bool,
        hist_param: Optional[Sequence] = None,
        hist_score: Optional[Sequence[float]] = None,
        opti_score: Optional[float] = None,
        full_sample: Optional[Sequence] = None,
        track_gen: Optional[Sequence[int]] = None,
        all_scores: Optional[Sequence] = None,
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
        self._full_sample = full_sample
        self._track_gen = track_gen
        self._all_scores = all_scores

    @property
    def full_sample(self):
        return self._full_sample

    @property
    def track_gen(self):
        return self._track_gen

    @property
    def all_scores(self):
        return self._all_scores

    def save(self, name: str, path: str = ".", overwrite: bool = True) -> str:
        acc_path = super().save(name, path, overwrite)
        if self._full_sample is not None:
            np.savetxt(
                os.path.join(acc_path, "full_sample.csv"), np.asarray(self._full_sample)
            )
        if self._track_gen is not None:
            np.savetxt(
                os.path.join(acc_path, "track_gen.csv"), np.asarray(self._track_gen)
            )
        if self._all_scores is not None:
            np.savetxt(
                os.path.join(acc_path, "all_scores.csv"), np.asarray(self._all_scores)
            )
        return acc_path


def construct_prior(sample: np.ndarray, frac_sparse: float = 0.0) -> np.ndarray:
    """
    Construct a gaussian distribution from a sample.
    Covariance of constructed distribution is diagonal.

    The output is a ProbaParam numpy.ndarray which can be passed to appropriate
    TensorizedGaussianMap.

    Args:
        sample: a sample from a (sparsified) gaussian distribution.
        frac_sparse: fraction of features which are set to their means.
    """
    means = np.apply_along_axis(np.mean, 0, sample)
    devs = np.apply_along_axis(np.std, 0, sample) / np.sqrt(1 - frac_sparse)
    return np.array([means, devs])


def iter_prior(
    score_fun: Callable[[np.ndarray], float],
    ini_prior_param: np.ndarray,
    gen_per_step: Union[int, Sequence[int]] = 100,
    chain_length: int = 1,
    keep: int = 250,
    frac_sparse: float = 0.0,
    parallel: bool = True,
    vectorized: bool = False,
    interactive: bool = True,
    silent: bool = True,
) -> OptimResultPriorIter:
    """
    Prior iteration joint Calibration and Uncertainty quantification routine.

    This code is adapted from a technique described in https://doi.org/10.1002/bit.28156

    Given a score function and a starting prior as a tensorized gaussian distribution, updates the
    prior by iteratively:
        - drawing samples from it,
        - fiting a new distribution from the k-best samples found so far

    The probability distribution is fitted by computing the mean and standard deviation of the best
    parameters, keeping "keep" of them. If less than keep samples where drawn so far, the prior is
    not updated.

    Args:
        score_fun: a scoring function
        ini_prior: the initial proba parameter. Passed to a TensorizedGaussianDistrib object
            (fist element are the means, second element the standard deviation)
        gen_per_step: number of samples generated at each step
        chain_length: number of iterations
        keep: number of samples kept to prepare the next generation
        frac_sparse: fraction of features set to their mean
        parallel: should score_fun calls be parallelized? Default is True, suited to non
            negligeable computation time in score_fun
        interactive: should the print be interactive?
        silent: should there be any print?

    Outputs an OptimResultPriorIter object which is assumed not to have converged.
    """
    prior_param = np.array(ini_prior_param)
    out_shape = prior_param.shape[1:]

    proba_map = TensorizedGaussianMap(sample_shape=out_shape)

    if isinstance(gen_per_step, int):
        gen_per_step = np.full(chain_length, gen_per_step)  # type: ignore
    else:
        chain_length = len(gen_per_step)

    hist_macro = AccuSampleVal(
        sample_shape=proba_map.proba_param_shape, n_tot=chain_length
    )

    tot_calls = np.sum(gen_per_step)
    accu_score = np.full(tot_calls, np.inf)
    accu_sample = np.zeros((tot_calls,) + out_shape)

    filled = 0
    m_score = np.inf

    # Set up sample generation tracker
    track_gen: list[int] = []

    for i, n_sample in enumerate(gen_per_step):  # type: ignore
        # Print information on last generation
        if interactive:
            sys.stdout.write("\r")
            sys.stdout.write(
                "Step %i/%i (mean score: %f)" % (i + 1, chain_length, m_score)
            )
            sys.stdout.flush()
        else:
            blab(
                silent,
                "Step %i/%i (mean score: %f)" % (i + 1, chain_length, m_score),
            )

        # Start loop properly
        prior = proba_map(prior_param)
        sample = prior(n_sample)

        # Sparsify prior
        if frac_sparse > 0:
            for samp in sample:
                choose_k = np.random.choice(len(samp), int(frac_sparse * len(samp)))
                samp[choose_k] = prior.means[choose_k]  # type: ignore

        if vectorized:
            scores = score_fun(sample)
        else:
            scores = par_eval(score_fun, sample, parallel=parallel)  # type: ignore

        accu_sample[filled : (filled + n_sample)] = np.array(sample)
        accu_score[filled : (filled + n_sample)] = np.array(scores)
        track_gen.extend(np.full(n_sample, i))
        filled += n_sample

        sorter = accu_score[:filled].argsort()
        accu_score[:filled] = accu_score[sorter]
        accu_sample[:filled] = accu_sample[sorter]
        track_gen = [track_gen[i] for i in sorter]

        best_sample = accu_sample[: min(filled, keep)]
        m_score = np.mean(scores)  # type: ignore

        if filled >= keep:
            prior_param = construct_prior(best_sample, frac_sparse)

        hist_macro.add1(sample=prior_param, val=m_score)

    blab(silent, "")

    return OptimResultPriorIter(
        opti_param=prior_param,
        converged=False,
        hist_param=hist_macro.sample(),
        hist_score=hist_macro.vals(),
        full_sample=accu_sample,  # type: ignore
        track_gen=track_gen,
        all_scores=accu_score,  # type: ignore
    )


def iter_prior_bayes(
    score_fun: Callable[[np.ndarray], float],
    prior_param: np.ndarray,
    temperature: float = 0.0,
    post_param: Optional[np.ndarray] = None,
    gen_per_step: Union[int, Sequence[int]] = 100,
    chain_length: int = 1,
    keep: int = 250,
    frac_sparse: float = 0.0,
    stop_tol: float = 0.0,
    parallel: bool = True,
    vectorized: bool = False,
    interactive: bool = True,
    silent: bool = True,
) -> OptimResultPriorIter:
    """
    Prior iteration joint Calibration and Uncertainty quantification routine.
    This is mainly thought as a jump-start procedure for pac-bayes minimisation

    Given a score function and a starting prior as a tensorized gaussian distribution,
    updates the prior by iteratively:
        - drawing samples from it,
        - fiting a new distribution from the k-best samples found so far
    Stops if the PAC-Bayes objective stops decreasing.

    The probability distribution is fitted by computing the mean and standard deviation of the best
    parameters, keeping "keep" of them. If less than keep samples where drawn so far, the prior is
    not updated.

    Args:
        score_fun: scoring function
        prior_param: prior parameter for a tensorized gaussian distribution
        temperature: PAC-Bayesian temperature
        post_param: optional initial posterior parameter for a tensorized gaussian distribution
    Secondary arguments:
        gen_per_step: number of samples generated at each step
        chain_length: number of iterations
        keep: number of samples kept to prepare the next generation
        frac_sparse: fraction of features set to their mean
        parallel: should score_fun calls be parallelized? Default is True, suited to non
            negligeable computation time in score_fun
        interactive: should the print be interactive?
        silent: should there be any print?

    Outputs an OptimResultPriorIter object which is assumed not to have converged.
    """

    # Convert to np.array for safety's sake
    prior_param = np.array(prior_param)
    out_shape = prior_param.shape[1:]

    if post_param is None:
        post_param = prior_param.copy()
    else:
        post_param = np.array(post_param).copy()

    proba_map = TensorizedGaussianMap(sample_shape=out_shape)

    if isinstance(gen_per_step, int):
        gen_per_step = np.full(chain_length, gen_per_step)  # type: ignore
    else:
        chain_length = len(gen_per_step)

    hist_macro = AccuSampleVal(
        sample_shape=proba_map.proba_param_shape, n_tot=chain_length
    )

    tot_calls = np.sum(gen_per_step)
    accu_score = np.full(tot_calls, np.inf)
    accu_sample = np.zeros((tot_calls,) + out_shape)

    filled = 0
    m_score = np.inf
    kl = 0.0

    # Set up sample generation tracker
    track_gen: list[int] = []
    converged = False

    # tol = norm().ppf(stop_tol)
    V_prev = np.inf

    i = 0
    while (i < chain_length) & (not converged):
        n_sample = gen_per_step[i]  # type: ignore
        # Print information on last generation
        if interactive:
            sys.stdout.write("\r")
            sys.stdout.write(
                "Step %i/%i (mean score: %f, kl : %f)"
                % (i + 1, chain_length, m_score, kl)
            )
            sys.stdout.flush()
        else:
            blab(
                silent,
                "Step %i/%i (mean score: %f, kl: %f)"
                % (i + 1, chain_length, m_score, kl),
            )

        # Start loop properly
        kl = proba_map.kl(post_param, prior_param)
        post = proba_map(post_param)
        sample = post(n_sample)

        # Sparsify prior
        if frac_sparse > 0:
            for samp in sample:
                choose_k = np.random.choice(len(samp), int(frac_sparse * len(samp)))
                samp[choose_k] = post_param[0][choose_k]

        if vectorized:
            scores = score_fun(sample)
        else:
            scores = par_eval(score_fun, sample, parallel=parallel)  # type: ignore

        accu_sample[filled : (filled + n_sample)] = np.array(sample)
        accu_score[filled : (filled + n_sample)] = np.array(scores)
        track_gen.extend(np.full(n_sample, i))
        filled += n_sample

        sorter = accu_score[:filled].argsort()
        accu_score[:filled] = accu_score[sorter]
        accu_sample[:filled] = accu_sample[sorter]
        track_gen = [track_gen[i] for i in sorter]

        best_sample = accu_sample[: min(filled, keep)]
        m_score = np.mean(scores)  # type: ignore

        V = m_score + temperature * kl

        if V < V_prev + stop_tol:
            if filled >= keep:
                post_param = construct_prior(best_sample, frac_sparse)

            hist_macro.add1(sample=prior_param, val=m_score)
            converged = False
            V_prev = V
        else:
            print(f"Converged (mean: {m_score}, kl: {kl})")
            converged = True

        i += 1
    blab(silent, "")

    return OptimResultPriorIter(
        opti_param=post_param,
        converged=False,
        hist_param=hist_macro.sample(),
        hist_score=hist_macro.vals(),
        full_sample=accu_sample,  # type: ignore
        track_gen=track_gen,
        all_scores=accu_score,  # type: ignore
        hyperparams={"temperature": temperature},
    )
