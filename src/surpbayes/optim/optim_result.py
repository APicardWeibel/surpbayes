"""
Optimisation result classes
"""

import os
from typing import Callable, Optional, Sequence

import dill
import numpy as np
from surpbayes.misc import par_eval


class OptimResult:
    """
    Class for output of optimization routines

    This class functions as an organized storage of optimisation related variables. These include
    - opti_param, the parameter returned by the optimisation routine
    - converged, whether the optimisation routine assumes convergence
    - opti_score, the score achieved by the optimisation routine (Optional)
    - hist_param, the list of parameters in the optimisation route (Optional)
    - hist_score, the scores of the parameters in hist_param (Optional)
    - full_evals, the full evaluations of x_i, S(x_i) generated during optimisation (Optional)
    - hyperparams, the hyperparameters used for the optimisation procedure (Optional)

    Note on saving
    Hyperparams are saved using dill, as non standard python hyperparameters could be provided.
    """

    class_name = "OptimResult"

    def __init__(
        self,
        opti_param,
        converged: bool,
        opti_score: Optional[float] = None,
        hist_param: Optional[Sequence] = None,
        hist_score: Optional[Sequence[float]] = None,
        hyperparams: Optional[dict] = None,
    ):
        """Initialize class from attributes values"""

        self._opti_param = opti_param
        self._converged = converged
        self._opti_score = opti_score
        self._hist_param = hist_param
        self._hist_score = hist_score
        self._hyperparams = hyperparams

    @property
    def opti_param(self):
        """Optimal parameter found during the optimisation process"""
        return self._opti_param

    @property
    def converged(self):
        """Whether the optimisation process converged"""
        return self._converged

    @property
    def opti_score(self) -> float:
        """Optimal score found during the optimisation process"""
        return self._opti_score

    @property
    def hist_param(self):
        """Parameter history throughout the optimisation process"""
        return self._hist_param

    @property
    def hist_score(self):
        """Score history throughout the optimisation process"""
        return self._hist_score

    @property
    def hyperparams(self):
        """Hyper parameters of the optimisation process"""
        return self._hyperparams

    def convert(self, fun: Callable, vectorized: bool = False, parallel: bool = False):
        """
        Convert parameters logged in OptimResult object inplace

        If J o fun was optimized in order to optimize J, then converts the optimisation result for
        the optimisation of J (i.e. parameters are converted)
        """

        self._opti_param = fun(self._opti_param)

        if self._hist_param is not None:
            if vectorized:
                self._hist_param = fun(self._hist_param)
            else:
                self._hist_param = par_eval(fun, self._hist_param, parallel)

    def get_best_param(self):
        """Check history for lowest score found"""
        if (self._hist_param is None) or (self._hist_score is None):
            raise ValueError("Empty hist_param or hist_score attributes")
        return self._hist_param[np.argmin(self._hist_score)]

    def save(self, name: str, path: str = ".", overwrite: bool = True) -> str:
        """Save 'OptimResult' object to folder 'name' in 'path'."""
        if not os.path.isdir(path):
            raise ValueError(f"{path} should point to a folder")
        acc_path = os.path.join(path, name)
        os.makedirs(acc_path, exist_ok=overwrite)

        with open(
            os.path.join(acc_path, "opti_type.txt"), "w", encoding="utf-8"
        ) as file:
            file.write(self.class_name)

        np.savetxt(os.path.join(acc_path, "opti_param.csv"), self._opti_param)

        with open(
            os.path.join(acc_path, "converged.txt"), "w", encoding="utf-8"
        ) as file:
            file.write(str(int(self._converged)))

        if self._opti_score is not None:
            with open(
                os.path.join(acc_path, "opti_score.txt"), "w", encoding="utf-8"
            ) as file:
                file.write(str(self._opti_score))
        if self._hist_score is not None:
            np.savetxt(
                os.path.join(acc_path, "hist_score.csv"), np.array(self._hist_score)
            )
        if self._hist_param is not None:
            np.savetxt(
                os.path.join(acc_path, "hist_param.csv"), np.array(self._hist_param)
            )

        if self._hyperparams is not None:
            with open(os.path.join(acc_path, "hyperparams.dl"), "wb") as file:
                dill.dump(self._hyperparams, file)

        return acc_path

    def __repr__(self):
        if self._converged:
            conv_status = "Converged"
        else:
            conv_status = "Not converged"
        return "\n".join(
            [
                f"{self.class_name} object",
                f"Status: {conv_status}",
                f"Optimal score: {self._opti_score}",
                f"Optimal parameter: {self._opti_param}",
            ]
        )
