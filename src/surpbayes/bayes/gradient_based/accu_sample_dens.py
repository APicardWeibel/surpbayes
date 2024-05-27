""" 
AccuSampleValDens class

Inherited from AccuSampleVal, with added half_ldens information (log density with respect to the
distribution it was generated)
"""

import os
import warnings
from typing import Optional, Union

import numpy as np
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.misc import ShapeError, check_shape
from surpbayes.proba import Proba, ProbaMap


class AccuSampleValDens(AccuSampleVal):
    """
    Manages the low level history of a PAC Bayesian optimisation problem
    Inherited from AccuSampleVal class (added half_ldens information)

    Data can be accessed through methods
        sample,
        half_ldens,
        vals,
        gen_tracker
    which take as input a number of data (optional, if None returns all data)

    sample is a list of points x,
    vals the list of evaluations of the scoring function at x,
    half_ldens the half log densities for x wrt to the distribution from which x was generated,
    gen_tracker the information pointing from which distribution x was generated (latest generation
        is 0, -1 indicates that the sample point is not yet generated)

    Data is stored in np.ndarray of fixed sized defined at initialisation. These should never be
    accessed by the users as they can be misleading (data suppression is lazy, i.e. only a count of
    stored data is changed).

    Main method is grad_score, used for the PAC Bayesian optimisation problem with corrected
    weights.

    Note:
        Half log density information is used to efficiently recompute the density ratio with an
        unknown distribution.
    """

    # For saving/loading purpose
    accu_type = "AccuSampleValDens"

    def __init__(self, sample_shape: tuple, n_tot: int):
        super().__init__(sample_shape, n_tot)

        self._half_ldens = np.zeros(n_tot)

    def add(self, sample, half_ldens, vals) -> None:  # type: ignore # pylint: disable=W0221
        # Format input
        sample = np.array(sample)
        m = len(sample)

        # Check conforming inputs
        check_shape(sample, (m,) + self.sample_shape)

        if (len(half_ldens) != m) or (len(vals) != m):
            raise ShapeError(
                f"sample, hald_ldens and vals should have same length ({m}, {len(half_ldens)}, {len(vals)})"
            )
        n = self.n_filled

        # Check that full memory is not exceeded
        if (n + m) > self.n_tot:
            warnings.warn("Maximum number of data reached")
            m = self.n_tot - n

        # Add information to memory
        self._sample[n : (n + m)] = sample[:m]
        self._half_ldens[n : (n + m)] = half_ldens[:m]
        self._vals[n : (n + m)] = vals[:m]

        # Update generations
        self._gen_tracker[: (n + m)] += 1

        # Update filled memory size
        self._n_gen += 1
        self._n_filled = n + m

    def extend_memory(self, n_add: int) -> None:
        """Add n_add slots to the memory"""
        AccuSampleVal.extend_memory(self, n_add)
        half_ldens = np.zeros(self.n_tot)
        half_ldens[: self.n_filled] = self.half_ldens()
        self._half_ldens = half_ldens

    def half_ldens(self, k: Optional[int] = None) -> np.ndarray:
        """
        Clean look at the half log densities

        By default, outputs all half log densities logged.
        If 'k' is provided, the last 'k' half log densities logged are returned.
        """
        if k is None:
            init = 0
        else:
            init = max(0, self.n_filled - k)

        return self._half_ldens[init : self.n_filled]

    def corr_weights(
        self,
        proba: Proba,
        k: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Selects the k last parameters and return them along with the evaluations and the correct
        weight corrections.

        The resulting samples and weights can be used to estimate integrals through
        ..math::
            \mathbb{E}_{proba}[f(x)] \simeq 1/N \sum \omega_i f(x_i)
        This is integral estimation is unbiased (variance analysis is not straightforward). The sub
        sums for each generation are also unbiased (but they are correlated with one another).
        """
        if k is None:
            k = self.n_filled

        return (
            self.sample(k),
            self.vals(k),
            proba.log_dens(self.sample(k)) - self.half_ldens(k),
        )

    def grad_score(
        self,
        d_map: ProbaMap,
        param: np.ndarray,
        gen_weights: Optional[Union[list, dict]] = None,
        gen_decay: float = 0.0,
        k: Optional[int] = None,
    ) -> tuple[np.ndarray, float, float]:
        r"""
        Outputs the derivative and evaluation at param of
        ..math::
            J(param) = \sum_{g>0} J_g(param) \exp(- g * gen_decay) / \sum_{g>0} \exp(-g * gen_decay)

        where :math:`J_g` uses the sample :math:`S_g` from generation g generated from :math:`param_g`
        to estimate the mean through
        ..math::
            J_g(param) = \sum_{x \in S_g} score(x) \times \exp(log_dens(x, param) - log_dens(x, param_g)) / \lvert S_g \rvert

        The intuition being that if the distributions generating all parameters are similar, then
        it is beneficial to use the previous evaluations of the score function in order to reduce
        the variance of the derivative estimate.

        Note:
            if log_dens(x, param) - log_dens(x, param_g) is too high (
            i.e. the point x generated through distribution param_g is deemed much more likely to have been generated from
            param than param_g
            ), then problematic behaviour might happen, the impact of this single point becoming disproportionate.

        Args:
            d_map, the distribution map used in the PAC Bayesian optimisation problem
            param, the parameter at which the derivative is to be computed
            gen_weights, an optional list of weights specifying how each generation should be weighted (first element = latest generation)
            gen_decay, used if gen_weights is None.
                Controls speed of exponentially decaying given to generation k through
                    w_k = exp(-gen_decay * k).
                Default is 0 (no decay, all generation with same weight).
            k, controls maximum number of sample used. None amounts to all sample used.
        """
        # Construct current distribution
        proba = d_map(param)
        # Prepare log_dens_der function
        der_log = d_map.log_dens_der(param)

        # Obtain proper weight corrections for samples from previous generations
        sample, vals, log_dens = self.corr_weights(proba, k=k)

        # Set up weight given to a generation
        n_gen = self.n_gen
        if gen_weights is None:
            gen_weights = [np.exp(-gen_decay * i) for i in range(n_gen)]

        # Tackle case where gen_weights information passed is insufficient
        if len(gen_weights) < n_gen:
            warnings.warn(
                f"Missing information in gen_weights. Giving weight 0 to all generations further than {len(gen_weights)}"
            )
            gen_weights = list(gen_weights) + [
                0 for i in range(n_gen - len(gen_weights))
            ]

        # Prepare specific weight given to each sample
        gen_tracker = self.gen_tracker(k)
        count_per_gen = [np.sum(gen_tracker == i) for i in range(n_gen)]

        gen_factor = np.array(
            [gen_weights[gen] / count_per_gen[gen] for gen in gen_tracker]
        )
        gen_factor = gen_factor / np.sum(gen_factor)

        weights = np.exp(log_dens) * gen_factor
        weights = weights / np.sum(weights)

        # Compute mean value
        mean_val = np.sum(vals * weights)
        # Compute uncertainty using last generation only
        UQ_val0 = np.std(vals[gen_tracker == 0]) / np.sqrt(np.sum(gen_tracker == 0) - 2)

        # Compute estimation of mean score gradient
        grads = der_log(sample)
        grad = np.tensordot((vals - mean_val) * weights, grads, (0, 0))

        return grad, mean_val, UQ_val0

    def save(self, name: str, path: str = ".", overwrite: bool = False) -> str:
        """Save AccuSampleValDens object to folder 'name' in 'path'"""
        acc_path = super().save(name, path, overwrite)
        np.savetxt(os.path.join(acc_path, "half_ldens.csv"), self.half_ldens())
        return acc_path

    def load(self, path: str) -> None:
        """Load data on an empty AccuSampleValDens instance.
        Memory is extended to suit the data loaded.
        """
        if self.n_filled > 0:
            raise ValueError(
                "Can not load data on a non empty AccuSampleValDens instance"
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

        path_half_ldens = os.path.join(path, "half_ldens.csv")
        if not os.path.isfile(path_half_ldens):
            raise FileNotFoundError(f"{path_half_ldens} does not exist")

        vals = np.loadtxt(path_vals)
        sample = np.loadtxt(path_sample)
        gen = np.loadtxt(path_gen, dtype=int)
        half_ldens = np.loadtxt(path_half_ldens)

        n = len(vals)
        assert len(gen) == n
        to_add = max(0, n - self.n_tot)

        if to_add > 0:
            self.extend_memory(to_add)
        self.add(sample, half_ldens, vals)
        self._gen_tracker[:n] = gen
