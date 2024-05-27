""" In the case of exponential family, add a T field to accu_sample.

Increase strain on memory but avoids recomputing T values at each iteration.
"""
import json
import os
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from surpbayes.accu_xy import AccuSampleVal
from surpbayes.proba import ExponentialFamily, PreExpFamily


class AccuSampleValExp(AccuSampleVal):
    """
    Manages the low level history of a PAC Bayesian optimisation problem.

    Data can be accessed through methods
        sample (all SamplePoints generated),
        vals (the score of each  SamplePoint),
        gen_tracker (when was each  SamplePoint generated)
    which take as input a number of data (optional, if None returns all data)

    sample is a list of points x,
    vals the list of evaluations of the scoring function at x,
    half_ldens the half log densities for x wrt to the distribution from which x was generated,
    gen_tracker the information pointing from which distribution x was generated (latest generation is 0,
        -1 indicates that the sample point is not yet generated)

    Data is stored in np.ndarray of fixed sized defined at initialisation. These should never be
    accessed by the users as they can be misleading (data suppression is lazy, i.e. only a count of
    stored data is changed).

    It is possible to increase memory size through extend_memory method.
    """

    # For saving/loading purpose
    accu_type = "AccuSampleValExp"

    def __init__(
        self, sample_shape: tuple[int, ...], t_shape: tuple[int, ...], n_tot: int
    ):
        super().__init__(sample_shape, n_tot)
        self._ts = np.zeros((n_tot,) + t_shape)
        self.t_shape = t_shape

    def extend_memory(self, n_add: int) -> None:
        n_filled = self.n_filled
        super().extend_memory(n_add)

        ts = np.zeros((self.n_tot,) + self.t_shape)
        ts[:n_filled] = self.ts()
        self._ts = ts

    def add(  # type: ignore# pylint: disable=W0221
        self, sample: ArrayLike, vals: ArrayLike, ts: ArrayLike
    ) -> None:
        """
        Add a new generation to memory.
        """

        n = self.n_filled
        super().add(sample, vals)  # type: ignore
        self._ts[n : self.n_filled] = ts

    def ts(self, k: Optional[int] = None) -> np.ndarray:
        """Clean look at the sample evaluations"""

        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)

        return self._ts[init : self.n_filled]

    def save(self, name: str, path: str = ".", overwrite: bool = False) -> str:
        acc_path = super().save(name, path, overwrite)
        np.savetxt(os.path.join(acc_path, "ts.csv"), self.ts())
        with open(os.path.join(acc_path, "t_shape.json"), "w", encoding="utf8") as file:
            json.dump(self.t_shape, file)
        return acc_path

    def load(self, path: str) -> None:
        """Load data on an empty AccuSampleValExp instance.
        Memory is extended to suit the data loaded.
        """
        if self.n_filled > 0:
            raise ValueError(
                "Can not load data on a non empty AccuSampleValExp instance"
            )
        # Check that path exists
        if not os.path.isdir(path):
            raise ValueError(f"{path} should point to a folder")

        # Check that all files are present.
        path_vals = os.path.join(path, "vals.csv")
        if not os.path.isfile(path_vals):
            raise FileNotFoundError(f"{path_vals} does not exist")

        path_sample = os.path.join(path, "sample.csv")
        if not os.path.isfile(path_sample):
            raise FileNotFoundError(f"{path_sample} does not exist")

        path_gen = os.path.join(path, "gen.csv")
        if not os.path.isfile(path_gen):
            raise FileNotFoundError(f"{path_gen} does not exist")

        path_ts = os.path.join(path, "ts.csv")
        if not os.path.isfile(path_ts):
            raise FileNotFoundError(f"{path_ts} does not exist")

        vals = np.loadtxt(path_vals)
        sample = np.loadtxt(path_sample)
        gen = np.loadtxt(path_gen, dtype=int)
        ts = np.loadtxt(path_ts)

        n = len(vals)
        assert len(gen) == n
        to_add = max(0, n - self.n_tot)

        if to_add > 0:
            self.extend_memory(to_add)
        self.add(sample, vals, ts)
        self._gen_tracker[:n] = gen


def _add_T_data(
    accu_sample: AccuSampleVal, proba_map: Union[PreExpFamily, ExponentialFamily]
) -> AccuSampleValExp:
    accu_exp = AccuSampleValExp(
        sample_shape=accu_sample.sample_shape,
        t_shape=proba_map.t_shape,
        n_tot=accu_sample.n_tot,
    )
    sample = accu_sample.sample()
    accu_exp.add(sample, vals=accu_sample.vals(), ts=proba_map.T(sample))
    accu_exp._gen_tracker = accu_sample._gen_tracker  # pylint:disable=W0212
    return accu_exp
