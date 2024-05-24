""" Inherited class of OptimResultBayes for gradient based VI algorithm"""

from typing import Optional, Sequence

from surpbayes.accu_xy import AccuSampleVal
from surpbayes.bayes.hist_bayes import HistBayesLog
from surpbayes.bayes.optim_result_bayes import OptimResultBayes
from surpbayes.types import ProbaParam


class OptimResultBayesGB(OptimResultBayes):
    """
    Inherited from OptimResultBayes

    Added fields:
        - bin_log_bayes
    """

    class_name = "OptimResultBayesGB"

    def __init__(
        self,
        opti_param: ProbaParam,
        converged: bool,
        opti_score: float,
        hist_param: Sequence[ProbaParam],
        hist_score: Sequence[float],
        end_param: ProbaParam,
        log_bayes: HistBayesLog,
        bin_log_bayes: HistBayesLog,
        sample_val: AccuSampleVal,
        hyperparams: Optional[dict] = None,
    ):
        super().__init__(
            opti_param=opti_param,
            converged=converged,
            opti_score=opti_score,
            hist_param=hist_param,
            hist_score=hist_score,
            end_param=end_param,
            log_bayes=log_bayes,
            sample_val=sample_val,
            hyperparams=hyperparams,
        )
        self._bin_log_bayes = bin_log_bayes

    @property
    def bin_log_bayes(self):
        return self._bin_log_bayes

    def save(self, name: str, path: str = ".", overwrite: bool = True) -> str:
        """Save 'OptimResultBayesGB' object to folder 'name' in 'path'"""

        # Saving 'OptimResultBayes' attributes
        acc_path = super().save(name, path, overwrite=overwrite)

        # Saving additional attributes
        (self._bin_log_bayes).save(name="bin_log_bayes", path=acc_path, overwrite=overwrite)
        return acc_path
