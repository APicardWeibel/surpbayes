"""
History Management for Meta Learning environnements.
"""
import json
import os
from typing import Optional, Sequence

import numpy as np
from surpbayes.bayes.hist_vi import FullMemory, check_shape
from surpbayes.misc import prod
from surpbayes.types import ProbaParam


class HistMeta:
    """
    History Management class for Meta Learning environnements.

    Accumulators are prepared before

    Logged data can be accessed through methods
        meta_params (all MetaParam generated),
        meta_scores (the list of score for each MetaParam),
        task_scores (the list of list of scores for each task in the learning environnement)
    """

    def __init__(self, meta_param_shape: tuple[int, ...], n: int, n_task: int):
        self.meta_param_shape = meta_param_shape
        self._meta_params = np.zeros((n,) + meta_param_shape)
        self._meta_scores = np.zeros(n)
        self._task_scores = np.zeros((n, n_task))

        # Specify memory size and amount filled
        self.n_filled = 0
        self.n = n
        self.n_task = n_task

    def _full(self):
        return self.n_filled == self.n

    def add(
        self,
        proba_pars: Sequence[ProbaParam],
        meta_scores: Sequence[float],
        task_scores: Sequence[Sequence[float]],
    ) -> None:
        """
        Store multiple new information in the history
        """
        n = len(proba_pars)

        _proba_pars = np.array(proba_pars)
        check_shape(_proba_pars, (n,) + self.meta_param_shape)

        _meta_scores = np.array(meta_scores)
        check_shape(_meta_scores, (n,))

        _task_scores = np.array(task_scores)
        check_shape(_task_scores, (n, self.n_task))

        n0 = self.n_filled

        if self._full():
            raise FullMemory("Already full")
        if n + n0 > self.n:
            raise Warning(f"Too much data is passed. Only storing first {self.n - n0}.")

        n = min(n, self.n - n0)

        self._meta_params[n0 : (n0 + n)] = proba_pars
        self._meta_scores[n0 : (n0 + n)] = meta_scores
        self._task_scores[n0 : (n0 + n)] = task_scores

        self.n_filled = self.n_filled + n

    def add1(
        self, proba_par: ProbaParam, meta_score: float, task_score: Sequence[float]
    ) -> None:
        """
        Store new information in the history. Similar to add, but does not expect list like elements.
        """
        if self._full():
            raise FullMemory("Already full")

        n = self.n_filled
        self._meta_params[n] = proba_par
        self._meta_scores[n] = meta_score
        self._task_scores[n] = task_score

        self.n_filled += 1

    def meta_params(self, k: Optional[int] = None) -> np.ndarray:
        """
        Clean look at the meta parameters

        By default, outputs all meta parameters logged.
        If 'k' is provided, the last 'k' meta parameters logged are returned.
        """
        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)
        return self._meta_params[init : self.n_filled]

    def meta_scores(self, k: Optional[int] = None) -> np.ndarray:
        """
        Clean look at the meta scores

        By default, outputs all meta scores logged.
        If 'k' is provided, the last 'k' meta scores logged are returned.
        """
        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)
        return self._meta_scores[init : self.n_filled]

    def task_scores(self, k: Optional[int] = None) -> np.ndarray:
        """
        Clean look at the task scores

        By default, outputs all task scores logged.
        If 'k' is provided, the last 'k' task scores list logged are returned.
        """
        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)
        return self._task_scores[init : self.n_filled]

    def get(self, k: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Outputs the description of the last k elements added to the memory
        """
        return self.meta_params(k), self.meta_scores(k), self.task_scores(k)

    def extend_memory(self, n_add: int = 1):
        """Add n_add memory slot to the AccuSampleVal object"""
        self._meta_params = np.concatenate(
            [self._meta_params, np.zeros((n_add,) + self.meta_param_shape)],
            0,
        )
        self._meta_scores = np.concatenate(
            [self._meta_scores, np.zeros(n_add)],
            0,
        )

        self._task_scores = np.concatenate(
            [self._task_scores, np.zeros((n_add, self.n_task))],
            0,
        )

        self.n = self.n + n_add

    def save(self, name: str, path: str = ".", overwrite: bool = False) -> str:
        """
        Save AccuSampleVal object to folder 'name' situated at 'path' (default to working folder)
        """
        if not os.path.isdir(path):
            raise ValueError(f"{path} should point to a folder")
        acc_path = os.path.join(path, name)
        os.makedirs(acc_path, exist_ok=overwrite)

        np.savetxt(
            os.path.join(acc_path, "meta_params.csv"),
            self.meta_params().reshape((self.n_filled, prod(self.meta_param_shape))),
        )

        np.savetxt(os.path.join(acc_path, "meta_scores.csv"), self.meta_scores())
        np.savetxt(os.path.join(acc_path, "task_scores.csv"), self.task_scores())

        with open(
            os.path.join(acc_path, "meta_param_shape.json"), "w", encoding="utf-8"
        ) as fjson:
            json.dump(self.meta_param_shape, fjson)

        return acc_path
