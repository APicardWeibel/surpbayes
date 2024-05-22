"""
Optimisation module

Main classes:
- Optimizer, dummy class for optimisation algorithms. Optimizer loops on
    "update" method until convergence or maximum number of iteration reached.
    "update" should be reimplemented for each specific routine.
- CMAOptimizer, class for CMA-ES optimisation routine
- MHOptimizer, class for a Metropolis-Hastings inspired optimisation routine
- OptimResult, main class for output of optimisation algorithm

Optimisation can be performed using "optim" function, with optimisation method
specified by "optimizer" argument.

Other:
A dichotomy solver for f(x) = y is also provided by "dichoto" function

FUTURE:
Optimisation routines optimise a np.ndarray input, and therefore expect a function
taking as first argument a np.ndarray input.

An interface 
"""

from surpbayes.optim.dichoto import dichoto
from surpbayes.optim.generative_optim import (CMAOptimizer, GenOptimizer,
                                         MHOptimizer, OptimResultCMA)
from surpbayes.optim.optim import optim
from surpbayes.optim.optim_result import OptimResult
from surpbayes.optim.optimizer import Optimizer
