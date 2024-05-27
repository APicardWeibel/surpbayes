import numpy as np
from surpbayes.bayes.surpac.pre_exp_solver import PreExpSPACS
from surpbayes.bayes.surpac.weighing import get_weights_mc_gauss
from surpbayes.misc import blab
from surpbayes.proba import Proba
from surpbayes.types import Samples


class GaussianSPACS(PreExpSPACS):
    """Bayesian Solver using SurPAC routine for Gaussian Family Maps

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
                    "Starting PAC-Bayes training",
                    "(SurPAC routine,",
                    "Gaussian variant)",
                ]
            ),
        )
