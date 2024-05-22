import os
from typing import Optional, Sequence

import numpy as np
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.bayes.hist_vi import HistVILog
from surpbayes.optim import OptimResult
from surpbayes.types import ProbaParam


class OptimResultVI(OptimResult):
    """
    Inherited from OptimResult.

    Added fields:
        - end_param
        - log_vi
        - sample_val
    """

    class_name = "OptimResultVI"

    def __init__(
        self,
        opti_param: ProbaParam,
        converged: bool,
        opti_score: float,
        hist_param: Sequence[ProbaParam],
        hist_score: Sequence[float],
        end_param: ProbaParam,
        log_vi: HistVILog,
        sample_val: AccuSampleVal,
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
        self._end_param = end_param
        self._log_vi = log_vi
        self._sample_val = sample_val

    @property
    def end_param(self):
        return self._end_param

    @property
    def log_vi(self):
        return self._log_vi

    @property
    def sample_val(self):
        return self._sample_val

    def save(self, name: str, path: str = ".", overwrite: bool = True) -> str:
        # Saving 'OptimResult' attributes
        acc_path = super().save(name, path, overwrite=overwrite)

        # Saving additional attributes
        np.savetxt(os.path.join(acc_path, "end_param.csv"), np.array(self.end_param))
        (self._sample_val).save(
            name="sample_val", path=acc_path, overwrite=overwrite
        )  # mypy: ignore-errors
        (self._log_vi).save(name="log_vi", path=acc_path, overwrite=overwrite)
        return acc_path
