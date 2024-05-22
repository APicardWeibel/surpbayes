"""Load optim result module

load_optim_result loads an OptimResult inherited object from a folder, automatically detecting
which variant has been stored.


TODO:
Instead of passing a opti_type.txt information when saving,
pass a much more robust json or yaml file indicating all extra fields
required during the construction, how they are stored and how they should be
loaded
"""

import os

import dill
import numpy as np

from surpbayes.bayes import (
    OptimResultPriorIter,
    OptimResultVI,
    OptimResultVIGB,
    load_hist_vi,
)
from surpbayes.load_accu import load_accu_sample_val
from surpbayes.optim import OptimResult, OptimResultCMA


def load_optim_result(path) -> OptimResult:
    """Load an OptimResult (or OptimResult inherited) object from a folder"""

    # Assess type of OptimResult object
    with open(os.path.join(path, "opti_type.txt"), "r", encoding="utf-8") as file:
        opti_type = file.read()

    # Load basic attributes shared among all subclass
    with open(os.path.join(path, "converged.txt"), "r", encoding="utf-8") as file:
        converged = file.read() == "True"

    opti_param = np.loadtxt(os.path.join(path, "opti_param.csv"))

    path_opti_score = os.path.join(path, "opti_score.txt")
    if os.path.isfile(path_opti_score):
        with open(path_opti_score, "r", encoding="utf-8") as file:
            opti_score = float(file.read())
    else:
        opti_score = None

    path_hist_score = os.path.join(path, "hist_score.csv")
    if os.path.isfile(path_hist_score):
        hist_score = np.loadtxt(path_hist_score)
    else:
        hist_score = None

    path_hist_param = os.path.join(path, "hist_param.csv")
    if os.path.isfile(path_hist_param):
        hist_param = np.loadtxt(path_hist_param)
    else:
        hist_param = None

    path_hyperparams = os.path.join(path, "hyperparams.dl")
    if os.path.isfile(path_hyperparams):
        with open(path_hyperparams, "rb") as file:
            hyperparams = dill.load(file)
    else:
        hyperparams = None

    # Check type + output correct result
    if opti_type == "OptimResult":
        return OptimResult(
            opti_param, converged, opti_score, hist_param, hist_score, hyperparams  # type: ignore
        )

    if opti_type == "OptimResultCMA":
        path_full_evals = os.path.join(path, "full_evals")
        if os.path.isdir(path_full_evals):
            full_evals = load_accu_sample_val(path_full_evals)
        else:
            full_evals = None

        hist_cov = np.loadtxt(os.path.join(path, "hist_cov.csv"))
        hist_cov = hist_cov.reshape(
            (len(hist_param), hist_cov.shape[1], hist_cov.shape[1])  # type: ignore
        )

        return OptimResultCMA(
            opti_param=opti_param,
            converged=converged,
            opti_score=opti_score,
            hist_param=hist_param,  # type: ignore
            hist_score=hist_score,  # type: ignore
            hist_cov=hist_cov,  # type: ignore
            full_evals=full_evals,
            hyperparams=hyperparams,
        )

    if opti_type == "OptimResultPriorIter":
        path_full_sample = os.path.join(path, "full_sample.csv")
        if os.path.isfile(path_full_sample):
            full_sample = np.loadtxt(path_full_sample)
        else:
            full_sample = None

        path_track_gen = os.path.join(path, "track_gen.csv")
        if os.path.isfile(path_track_gen):
            track_gen = np.loadtxt(path_track_gen)
        else:
            track_gen = None

        path_all_scores = os.path.join(path, "all_scores.csv")
        if os.path.isfile(path_all_scores):
            all_scores = np.loadtxt(path_all_scores)
        else:
            all_scores = None

        return OptimResultPriorIter(
            opti_param=opti_param,
            converged=converged,
            hist_param=hist_param,  # type: ignore
            hist_score=hist_score,  # type: ignore
            opti_score=opti_score,
            full_sample=full_sample,  # type: ignore
            track_gen=track_gen,  # type: ignore
            all_scores=all_scores,  # type: ignore
            hyperparams=hyperparams,
        )

    if opti_type == "OptimResultVIGB":
        if opti_score is None:
            raise ValueError("OptimResultVIGB should have an opti_score file")
        end_param = np.loadtxt(os.path.join(path, "end_param.csv"))

        path_sample_val = os.path.join(path, "sample_val")
        sample_val = load_accu_sample_val(path_sample_val)

        log_vi = load_hist_vi(os.path.join(path, "log_vi"))
        bin_log_vi = load_hist_vi(os.path.join(path, "bin_log_vi"))

        return OptimResultVIGB(
            opti_param=opti_param,
            converged=converged,
            opti_score=opti_score,  # type: ignore
            hist_param=hist_param,  # type: ignore
            hist_score=hist_score,  # type: ignore
            end_param=end_param,
            log_vi=log_vi,
            bin_log_vi=bin_log_vi,
            sample_val=sample_val,
            hyperparams=hyperparams,
        )

    if (opti_type == "OptimResultVI") or (opti_type == "OptimResultVISA"):
        if opti_score is None:
            raise ValueError("OptimResultVIGB should have an opti_score file")

        end_param = np.loadtxt(os.path.join(path, "end_param.csv"))
        log_vi = load_hist_vi(os.path.join(path, "log_vi"))
        sample_val = load_accu_sample_val(os.path.join(path, "sample_val"))

        return OptimResultVI(
            opti_param=opti_param,
            converged=converged,
            opti_score=opti_score,  # type: ignore
            hist_param=hist_param,  # type: ignore
            hist_score=hist_score,  # type: ignore
            end_param=end_param,
            log_vi=log_vi,
            sample_val=sample_val,
            hyperparams=hyperparams,
        )

    raise ValueError(f"opti_type value unknown ({opti_type})")
