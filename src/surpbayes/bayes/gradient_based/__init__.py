r"""
Optimise Catoni's bound through an approximated gradient descent procedure.

Contrary to the algorithm in the 'score_approx' module, the routines implemented here can be used
to optimise Catoni's bound for any 'ProbaMap.'

Warning:
- The algorithm seems to have stability issues, which prevent it from converging to the correct
distribution when the temperature is small (i.e. high learning rate).
"""
from .accu_sample_dens import AccuSampleValDens
from .gradient_based_solver import GradientBasedBayesSolver
from .knn_solver import KNNBayesSolver
from .optim_result_bayes_gb import OptimResultBayesGB
