"""
AccuSampleVal class.

Meant to be used to collect :math:`(x, f(x))` evaluations.

New evaluations can be added using the add method. Data suppression is performed lazily.

The generation at which each data was added is also stored.
Data can be saved to .csv files using the save method.

Loading AccuSampleVal object is done using the 'load_accu_sample_val' function in 'load_accu'
submodule. 'load_accu_sample_val' checks if the saved data does not describe a child class of
AccuSampleVal with added fields, and automatically loads the appropriate child class.
"""

import json
import os
import warnings
from typing import Callable, Optional

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from surpbayes.misc import check_shape, par_eval, prod
from surpbayes.types import SamplePoint, Samples


class AccuSampleVal:
    """
    Store evaluations of a function.

    Data can be accessed through methods
        sample (all SamplePoints generated),
        vals (the score of each  SamplePoint),
        gen_tracker (when was each  SamplePoint generated)
    which take as input a number of data (optional, if None returns all data)

    sample is a list of points x,
    vals the list of evaluations of the scoring function at x,
    half_ldens the half log densities for x wrt to the distribution from which x was generated,
    gen_tracker the information pointing from which distribution x was generated (latest generation
    is 0, -1 indicates that the sample point is not yet generated)

    Data is stored in np.ndarray of fixed sized defined at initialisation. These should never be
    accessed by the users as they can be misleading (data suppression is lazy, i.e. only a count of
    stored data is changed).

    Memory size can be extended through extend_memory method.
    """

    # For saving/loading purpose
    accu_type = "AccuSampleVal"

    def __init__(self, sample_shape: tuple[int, ...], n_tot: int):
        self.sample_shape = sample_shape

        self._sample = np.zeros((n_tot,) + sample_shape)
        self._vals = np.zeros(n_tot)
        self._gen_tracker = np.full(n_tot, -1)

        self._n_gen: int = 0

        self._n_filled: int = 0
        self._n_tot: int = n_tot

    @property
    def n_gen(self):
        """Number of generation (i.e. data adding evenements)"""
        return self._n_gen

    @property
    def n_filled(self):
        """Number of memory slots filled"""
        return self._n_filled

    @property
    def n_tot(self):
        """Total number of memory slots prepared"""
        return self._n_tot

    def extend_memory(self, n_add: int) -> None:
        """Add n_add memory slot to the AccuSampleVal object"""
        n_tot = self._n_tot + n_add
        n_filled = self._n_filled

        sample = np.zeros((n_tot,) + self.sample_shape)
        vals = np.zeros(n_tot)
        gen_tracker = np.full(n_tot, -1)

        sample[:n_filled] = self.sample()
        vals[:n_filled] = self.vals()
        gen_tracker[:n_filled] = self.gen_tracker()

        self._sample = sample
        self._vals = vals
        self._gen_tracker = gen_tracker

        self._n_tot = n_tot

    def n_remain(self):
        """Return number of remaining slots in the memory"""
        return self._n_tot - self._n_filled

    def add(self, sample: Samples, vals: np.ndarray) -> None:
        """
        Add a new generation to memory.
        """
        sample = np.asarray(sample)
        m = len(sample)

        check_shape(sample, (m,) + self.sample_shape)

        n = self._n_filled

        if (n + m) > self._n_tot:
            warnings.warn("Maximum number of data reached")
            m = self._n_tot - n

        self._sample[n : (n + m)] = sample[:m]
        self._vals[n : (n + m)] = vals[:m]

        self._gen_tracker[: (n + m)] += 1

        self._n_gen += 1
        self._n_filled = n + m

    def add1(self, sample: SamplePoint, val: float) -> None:
        """
        Add a new point to memory
        """
        sample = np.asarray(sample)

        check_shape(sample, shape_exp=self.sample_shape)

        n = self._n_filled

        if n < self._n_tot:
            self._sample[n] = sample
            self._vals[n] = val

            self._gen_tracker[: (n + 1)] += 1

            self._n_gen += 1
            self._n_filled = n + 1
        else:
            warnings.warn("Maximum number of data reached")

    def suppr(self, k: int):
        """Deletes the last k entries in the memory (lazy delete)"""
        self._n_filled = max(0, self._n_filled - k)

    def suppr_gen(self, K: int):
        """Deletes the last K generations in the memory (lazy delete)"""
        # Backup gen_tracker
        gen_tracker = self._gen_tracker.copy()
        gen_tracker = np.clip(gen_tracker - K, a_min=-1, a_max=None)

        self._n_gen = max(0, self._n_gen - K)
        self._n_filled = np.sum(gen_tracker >= 0, dtype=int)
        self._gen_tracker = gen_tracker

    def sample(self, k: Optional[int] = None) -> Samples:
        """
        Clean look at the samples

        By default, outputs all samples logged.
        If 'k' is provided, the last 'k' samples logged are returned.
        """
        if k is None:
            init = 0
        else:
            init = max(0, self._n_filled - k)

        return self._sample[init : self._n_filled]

    def vals(self, k: Optional[int] = None) -> np.ndarray:
        """
        Clean look at the sample evaluations

        By default, outputs all vals logged.
        If 'k' is provided, the last 'k' vals logged are returned.
        """

        if k is None:
            init = 0
        else:
            init = max(0, self._n_filled - k)

        return self._vals[init : self._n_filled]

    def gen_tracker(self, k: Optional[int] = None):
        """
        Clean look at the sample generations

        By default, outputs all sample generations logged.
        If 'k' is provided, the last 'k' sample generations logged are returned.
        """

        if k is None:
            init = 0
        else:
            init = max(0, self._n_filled - k)

        return self._gen_tracker[init : self._n_filled]

    def knn(self, k, *args, **kwargs):
        """
        Future:
            Using KNeighborsRegressor.score could be useful to choose which values should be
                evaluated
        """

        knn = KNeighborsRegressor(*args, **kwargs)
        knn.fit(self.sample(k), self.vals(k))

        return knn.predict

    def __repr__(self):
        return f"{self.accu_type} object with {self._n_filled} / {self._n_tot} evaluations filled"

    def convert(
        self, fun: Callable, vectorized: bool = False, parallel: bool = False
    ) -> None:
        """
        Convert inplace the sample of a log eval object.

        Both the "_sample" and "sample_shape" attributes are modified.

        Args:
            fun: the conversion function
            vectorized: states if fun is vectorized
            parallel: states if
        """
        if self._n_filled == 0:
            raise ValueError("Can not convert empty AccuSampleVal")
        if vectorized:
            converted_sample = fun(self.sample())
        else:
            converted_sample = np.array(par_eval(fun, self.sample(), parallel))

        new_shape = converted_sample.shape[1:]

        self._sample = np.concatenate(
            [converted_sample, np.zeros((self._n_tot - self._n_filled,) + new_shape)], 0
        )
        self.sample_shape = new_shape

    def save(self, name: str, path: str = ".", overwrite: bool = False) -> str:
        """
        Save AccuSampleVal object to folder 'name' situated at 'path' (default to working folder)
        """
        if not os.path.isdir(path):
            raise ValueError(f"{path} should point to a folder")
        acc_path = os.path.join(path, name)
        os.makedirs(acc_path, exist_ok=overwrite)

        # Save accu_type information (for loading)
        with open(
            os.path.join(acc_path, "acc_type.txt"), "w", encoding="utf-8"
        ) as file:
            file.write(self.accu_type)

        np.savetxt(os.path.join(acc_path, "vals.csv"), self.vals())
        np.savetxt(
            os.path.join(acc_path, "sample.csv"),
            self.sample().reshape((self._n_filled, prod(self.sample_shape))),
        )
        np.savetxt(os.path.join(acc_path, "gen.csv"), self.gen_tracker())
        with open(
            os.path.join(acc_path, "sample_shape.json"), "w", encoding="utf-8"
        ) as fjson:
            json.dump(self.sample_shape, fjson)

        return acc_path

    def load(self, path: str) -> None:
        """Load data on an empty AccuSampleVal instance.
        Memory is extended to suit the data loaded.
        """
        if self._n_filled > 0:
            raise ValueError("Can not load data on a non empty AccuSampleVal instance")
        # Check that path exists
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} should point to a folder")

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

        vals = np.loadtxt(path_vals)
        sample = np.loadtxt(path_sample)
        gen = np.loadtxt(path_gen, dtype=int)

        n = len(vals)
        if not len(gen) == n:
            raise ValueError(
                f"Number of generations is not equal to number of values passed ({len(gen)}, {n})"
            )

        to_add = max(0, n - self._n_tot)

        if to_add > 0:
            self.extend_memory(to_add)
        self.add(sample, vals)
        self._gen_tracker[:n] = gen

    def downgrade(self):
        """Downgrade a subclass of AccuSampleVal back to AccuSampleVal"""
        accu = AccuSampleVal(self.sample_shape, self._n_tot)
        accu.add(self.sample(), self.vals())
        accu._gen_tracker = self._gen_tracker  # pylint: disable=W0212

        return accu
