import numpy as np
from surpbayes.bayes.surpac.pre_exp_solver import PreExpSPACS
from surpbayes.bayes.surpac.weighing import get_weights_mc_gauss
# from multiprocess import Pool  # pylint: disable=E0611
from surpbayes.misc import blab
from surpbayes.proba import Proba
from surpbayes.types import Samples


class GaussianSPACS(PreExpSPACS):
    """Bayesian Solver using Score approximation routine for Gaussian Family Maps

    Differs from routine for standard PreExpFamily by the weighing technique (covariance matrix
    of the proba used.)
    """

    def weigh(self, proba: Proba, samples: Samples, n_sample_estim: int) -> np.ndarray:
        return get_weights_mc_gauss(
            proba, samples, n_sample_estim, k_neighbors=self.k_neighbors
        )

    def msg_begin_calib(self) -> None:
        blab(
            self.silent,
            " ".join(
                [
                    "Starting Bayesian calibration",
                    "(Score Approximation routine,",
                    "Gaussian variant)",
                ]
            ),
        )
