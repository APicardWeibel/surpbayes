""" Bayesian inspired algorithm for joint calibration and uncertainty quantification
Main function:
- pacbayes_minimize, designed to minimize Catoni's bound
    (see https://doi.org/10.48550/arXiv.2110.11216)
"""
from .bayes_solver import BayesSolver
from .gradient_based import (AccuSampleValDens, GradientBasedBayesSolver,
                             KNNBayesSolver, OptimResultBayesGB)
from .hist_bayes import HistBayesLog, load_hist_bayes
from .optim_result_bayes import OptimResultBayes
from .pacbayes_minimize import infer_pb_routine, pacbayes_minimize
from .score_approx import (AccuSampleValExp, GaussianSABS, PreExpSABS,
                           ScoreApproxBayesSolver)
