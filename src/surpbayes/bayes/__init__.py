""" Bayesian inspired algorithm for joint calibration and uncertainty quantification
Main functions:
- iter_prior, inspired by A. Leurent and R. Moscoviz (https://doi.org/10.1002/bit.28156)
- iter_prior_bayes, adaptation of iter_prior to the context of PAC-Bayes bound minimisation
- pacbayes_minimize, based on Catoni's bound (see https://doi.org/10.48550/arXiv.2110.11216)
"""
from .bayes_solver import BayesSolver
from .gradient_based import (AccuSampleValDens, GradientBasedBayesSolver,
                             KNNBayesSolver, OptimResultBayesGB)
from .hist_bayes import HistBayesLog, load_hist_bayes
from .iter_prior import OptimResultPriorIter, iter_prior, iter_prior_bayes
from .optim_result_bayes import OptimResultBayes
from .pacbayes_minimize import infer_pb_routine, pacbayes_minimize
from .score_approx import (AccuSampleValExp, GaussianSABS, PreExpSABS,
                           ScoreApproxBayesSolver)
