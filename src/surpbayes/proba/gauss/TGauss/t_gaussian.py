import warnings
from typing import Optional, Union

import numpy as np
from surpbayes.misc import ShapeError, prod
from surpbayes.proba.gauss.Gauss import Gaussian
from surpbayes.proba.proba import Proba
from surpbayes.types import SamplePoint, Samples


class TensorizedGaussian(Proba):
    """
    Gaussian multivariate distribution with diagonal covariance matrix class.
    Inherited from Proba class. Constructed from means and standard deviations.

    As for Proba, it is assumed that the distribution is defined on np.ndarray like.
    The shape of a sample is defined by the shape of the mean. Stored information is
    flattened.

    TODO: check if this can not be inherited from Gaussian class
    (Currently not done since super call overwrites more efficient log_dens + gen implementations)
    """

    def __init__(
        self,
        means: np.ndarray,
        devs: np.ndarray,
        sample_shape: Optional[tuple[int, ...]] = None,
    ):
        """
        Constructs a gaussian distribution from means and standard deviations, assuming
        that the covariance is diagonal.
        The shape of the sample is determined by the shape of the means. The standard
        deviations is flattened.
        """

        # Start by passing means and infer shape from that. This is corrected later if sample_shape
        # argument is not None
        self._shaped_means = np.array(means)
        self._sample_shape = self._shaped_means.shape
        self._n_dim_shape = len(self._sample_shape)
        self._means = self._shaped_means.flatten()
        self._sample_size = prod(self._sample_shape)
        self._updt_sample_shape(sample_shape)
        self._updt_devs(devs)
        self._vects = np.eye(self._sample_size)

        def log_dens(samples: Samples) -> np.ndarray:
            samples = np.array(samples)
            pre_shape = samples.shape[: -self._n_dim_shape]
            samples = samples.reshape(pre_shape + (self._sample_size,))

            return (
                -0.5 * (((samples - self._means) / self._devs) ** 2).sum(-1)
                + self._renorm_const
            )

        def gen(n: int) -> Samples:
            return (
                self._means
                + self._devs * np.random.normal(0, 1, (n, self._sample_size))
            ).reshape((n,) + self._sample_shape)

        super().__init__(
            log_dens=log_dens, gen=gen, sample_shape=self._sample_shape, np_out=True
        )

    def _updt_sample_shape(self, sample_shape: Optional[tuple[int, ...]]):
        """Update sample shape safely.
        If passed sample_shape is None, pass"""
        if sample_shape is None:
            return None

        if prod(sample_shape) != self._sample_size:
            warnings.warn(
                f"""Size of means ({self._sample_size} and sample_shape ({sample_shape}) are not compatible.
                'sample_shape' is not updated.
                """
            )
            return None
        self._sample_shape = sample_shape
        self._shaped_means = self._means.reshape(self._sample_shape)
        self._n_dim_shape = len(self._sample_shape)

    @property
    def means(self):
        """Flat means of the distribution"""
        return self._means

    @means.setter
    def means(self, value):
        # Check value size
        nmeans = np.array(value).flatten()
        if len(nmeans) != self._sample_size:
            raise ValueError(
                f"Proposed new means is not of adequate size (expected {self._sample_size}, got shape ({nmeans.shape}))"
            )
        self._means = nmeans
        self._shaped_means = nmeans.reshape(self._sample_shape)

    @property
    def shaped_means(self):
        """Means of the probability distribution"""
        return self._shaped_means

    @shaped_means.setter
    def shaped_means(self, value):
        nsmeans = np.array(value)
        new_shape = nsmeans.shape
        if prod(new_shape) != self._sample_size:
            msg = " ".join(
                [
                    "Proposed new means is not of adequate size",
                    f"(expected {self._sample_size}, got shape ({new_shape}))",
                ]
            )
            raise ValueError(msg)
        self._means = nsmeans.flatten()
        self._updt_sample_shape(new_shape)

    @property
    def vals(self):
        return self._vals

    @property
    def devs(self):
        """Standard deviations"""
        return self._devs

    @devs.setter
    def devs(self, value):
        self._updt_devs(value)

    @property
    def sample_shape(self):
        return self._sample_shape

    @sample_shape.setter
    def sample_shape(self, value):
        self._updt_sample_shape(sample_shape=value)

    def _updt_devs(self, devs: np.ndarray) -> None:
        """Update standard deviations and related fields"""
        devs_arr = np.array(devs).flatten()
        if len(devs_arr) != self._sample_size:
            raise ShapeError("Means and standard deviations should have the same size")

        self._devs = devs_arr
        self._renorm_const = -0.5 * self._sample_size * np.log(2 * np.pi) - np.sum(
            np.log(devs)
        )
        # For compatibility with gaussian distributions
        self._vals = self._devs**2

    def __repr__(self):
        return str.join(
            "\n",
            [
                f"Means:\n{self._shaped_means}\n",
                f"Standard deviations:\n{self.devs.reshape(self.sample_shape)}",
            ],
        )

    def copy(self):
        return TensorizedGaussian(self.means.copy(), self.devs.copy())

    def as_Gaussian(self) -> Gaussian:
        """Return identical distribution as a full Gaussian distribution"""
        return Gaussian(
            means=self.means.copy(),
            cov=np.diag(self._vals),
            info={"vals": self._vals, "vects": np.eye(self._sample_size)},
        )

    def shift(self, shift: SamplePoint):
        """
        Transform the distribution of X to the distribution of X + shift
        """
        return TensorizedGaussian(
            means=self.means.reshape(self.sample_shape) + shift,
            devs=self.devs,
            sample_shape=self.sample_shape,
        )

    def contract(self, alpha: float):
        """
        Transform the distribution of X to the distribution of alpha * X

        Argument:
            alpha: a float
        """
        return TensorizedGaussian(
            means=self.means,
            devs=alpha * self.devs,
            sample_shape=self.sample_shape,
        )

    def lin_transform(self, mat: np.ndarray, shift: Union[float, SamplePoint] = 0.0):
        """
        Transform the distribution of X to the distribution of mat @ X + shift
        (where the @ denote a full tensor product rather than matrix product).

        Shift can be either a float or a np.array object of shape compatible with the matrix.
        """

        return self.as_Gaussian().lin_transform(mat=mat, shift=shift)

    def marginalize(self, indexes: list[int]):
        """
        Get the marginal distribution of (X_i)_{i in indexes} from distribution of X.

        Due to renormalization issues in the general case, this method is only possible
        for specific probability distributions such as Gaussian.

        The resulting distribution operates on 1D array.
        """
        new_means, new_devs = self.means.copy(), self.devs.copy()

        new_means = new_means[indexes]
        new_devs = new_devs[indexes]

        return TensorizedGaussian(new_means, new_devs, sample_shape=(len(new_means),))
