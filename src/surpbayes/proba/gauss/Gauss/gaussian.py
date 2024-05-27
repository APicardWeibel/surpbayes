"""
Subclass of Proba for Gaussian distributions

lin_transform, shift, contract, reshape, marginalize are overwritten.
"""

import warnings
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from surpbayes.misc import ShapeError, _get_pre_shape, prod
from surpbayes.proba.proba import Proba
from surpbayes.proba.warnings import NegativeCov
from surpbayes.types import SamplePoint, Samples


class Gaussian(Proba):
    """
    Gaussian multivariate distribution.
    Inherited from Proba class. Constructed from means and covariance matrix.

    Shape of samples:
        The shape of the samples can be specified from the sample_shape argument.
        If the sample_shape argument is None, this reverts to the shape of the means.
        If the sample_shape argument is incoherent with the shape of the means, throws a
            warning and uses the shape of the means.

        Regardless of the resulting sample_shape, the covariance must be a square matrix. If not,
        a warning is thrown and the covariance is reshaped if possible.
    """

    def __init__(
        self,
        means: ArrayLike,
        cov: ArrayLike,
        info: Optional[dict] = None,
        sample_shape: Optional[tuple[int, ...]] = None,
    ):
        """
        Constructs a gaussian distribution for mean and covariance matrix.

        Argument:
            means: means of the gaussian distribution.
            cov: covariance of the gaussian distribution.
            info: optional dictionary containing the eigenvalues and eigenvectors of the covariance
                (keys: 'vals', 'vects')
            sample_shape: optional tuple specifying the shape of the output. If not provided, use
                the shape of means.
        """

        # Force convert to np.ndarray
        self._shaped_means = np.array(means)

        self._sample_shape = self._shaped_means.shape
        self._means = self._shaped_means.flatten()
        self._sample_size = prod(self._sample_shape)  # Immutable argument
        self._updt_sample_shape(sample_shape)

        self._updt_cov(cov, info)

        # Define log_dens function for gaussian distribution
        def log_dens(samples: Samples) -> np.ndarray:
            samples = np.array(
                samples
            )  # Force convert array. Necessary due to transform behavior

            pre_shape = _get_pre_shape(samples, self._sample_shape)  # type: ignore

            # Right flatten of array
            samples = samples.reshape(pre_shape + (self._sample_size,))
            centered = samples - self._means
            # Using the fact that inv_cov is symmetric to use right multiplication
            dist = (centered * (centered @ self._inv_cov)).sum(-1)
            return -0.5 * dist + self._renorm_const

        def gen(n: int) -> Samples:
            # Slightly more efficient than np.random.multivariate, notably if dim is large
            return (
                self._means
                + np.random.normal(0, 1, (n, self._sample_size)) @ self._half_cov
            ).reshape(
                (n,) + self._sample_shape  # type: ignore
            )

        super().__init__(
            log_dens=log_dens, gen=gen, sample_shape=sample_shape, np_out=True
        )

    def _updt_sample_shape(self, sample_shape: Optional[tuple[int, ...]]) -> None:

        # Check compatibility of means and sample_shape
        if sample_shape is None:
            return None

        if prod(sample_shape) != self._sample_size:
            warnings.warn(
                """ 'sample_shape' information passed is incompatible with means.
                'sample_shape' is not updated."""
            )
            return None

        self._sample_shape = sample_shape
        self._shaped_means = self._means.reshape(self._sample_shape)

    def _updt_cov(self, new_cov, info=None):

        ncov_arr = np.array(new_cov)
        if (self._sample_size**2) != prod(ncov_arr.shape):
            raise ShapeError(
                f"Covariance shape ({ncov_arr.shape}) and means shape ({self._sample_shape}) are not compatible."
            )

        if ncov_arr.shape != (self._sample_size, self._sample_size):
            # Try finding covariance by reshaping
            warnings.warn("Passed covariance is not a square matrix. Reshaping.")
            ncov_arr = ncov_arr.reshape((self._sample_size, self._sample_size))

        # Check that covariance is almost symmetric and force symmetry
        if np.allclose(ncov_arr, ncov_arr.T):
            ncov_arr = (ncov_arr + ncov_arr.T) / 2
        else:
            raise ValueError(f"'cov' must be symmetric.\n'cov':\n {ncov_arr}")

        self._cov = ncov_arr

        # From now on, one can assume that both cov and means are well formatted
        if info is None:
            self._vals, self._vects = np.linalg.eigh(self._cov)  # type: ignore
        else:
            self._vals, self._vects = info["vals"], info["vects"]

        # Check for negative eigenvalues
        if self._vals[0] < 0:
            warnings.warn(
                "Covariance matrix had negative eigenvalues. Setting them to 0.",
                category=NegativeCov,
            )

        # Pre compute constants for log density function
        self._vals = np.maximum(self._vals, 0.0)
        self._singular = np.min(self._vals) == 0.0

        inv_vals = np.array([val**-1 if val > 0 else 0 for val in self._vals])
        self._inv_cov = (inv_vals * self._vects) @ self._vects.T

        self._renorm_const = -0.5 * self._sample_size * np.log(
            2 * np.pi
        ) - 0.5 * np.sum(np.log(self._vals))

        # Pre compute constant for gen function
        half_vals = self._vals**0.5
        self._half_cov = (half_vals * self._vects) @ self._vects.T

    @property
    def sample_shape(self):
        return self._sample_shape

    @sample_shape.setter
    def sample_shape(self, value: tuple[int, ...]):
        self._updt_sample_shape(value)

    @property
    def cov(self):
        """Covariance matrix"""
        return self._cov

    @cov.setter
    def cov(self, value):
        self._updt_cov(new_cov=value)

    @property
    def inv_cov(self):
        """Inverse of covariance matrix"""
        return self._inv_cov

    @property
    def vals(self):
        """Eigenvalues of the covariance"""
        return self._vals

    @property
    def vects(self):
        """Eigenvectors of the covariance"""
        return self._vects

    @property
    def singular(self):
        """Whether or not the covariance is singular"""
        return self._singular

    @property
    def means(self):
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

    def __repr__(self):
        return str.join(
            "\n",
            [
                "Gaussian Distribution",
                f"Mean: {self._shaped_means}",
                f"Covariance : {self._cov}",
            ],
        )

    def copy(self):
        """Copy of the Gaussian distribution object.
        Avoids dependency on the cov and means.
        """
        return Gaussian(
            means=self._means.copy(),
            cov=self._cov.copy(),
            sample_shape=self._sample_shape,
            info={"vals": self._vals.copy(), "vects": self._vects.copy()},
        )

    def reshape(self, new_shape: tuple[int, ...]):
        """
        Transforms the shape of the samples.

        If the distribution generates samples with each shapes of
            (n1, ..., nk),
        the distribution proba.reshape( (m1, ..., m \tilde{k})) will output samples of shape
            (m1, ..., m \tilde{k})
        IF n1 * ... * nk = m1 * ... * m \tilde{k} (else a ShapeMismatch exception is raised when
        trying to construct proba.reshape)

        Note:
            The new distribution will generate np.ndarray objects
        """
        return Gaussian(
            means=self._means,
            cov=self._cov,
            info={"vals": self._vals, "vects": self._vects},
            sample_shape=new_shape,
        )

    # flatten uses the .reshape method from the instance, no need to reimplement
    def contract(self, alpha: float):
        """
        Transform the distribution of X to the distribution of alpha * X

        Argument:
            alpha: a float
        """
        return Gaussian(
            means=alpha * self._means,
            cov=(alpha**2) * self._cov,  # type: ignore
            info={"vals": (alpha**2) * self._vals, "vects": self._vects},
            sample_shape=self._sample_shape,
        )

    def shift(self, shift: SamplePoint):
        """
        Transform the distribution of X to the distribution of X + shift
        """
        return Gaussian(
            means=self.shaped_means + shift,
            cov=self.cov,
            info={"vals": self._vals, "vects": self._vects},
            sample_shape=self._sample_shape,
        )

    def lin_transform(self, mat: np.ndarray, shift: Union[float, SamplePoint] = 0.0):
        """
        Transform the distribution of X to the distribution of mat @ X + shift
        (where the @ denote a full tensor product rather than matrix product).

        Shift can be either a float or a np.array object of shape compatible with the matrix.

        Dimension formating:
            If proba outputs samples of shape (n1, ..., nk), then the matrix should be shaped
                (m1, ..., m \tilde{k}, n1, ..., nk)
            with m1, ..., m \tilde k such that m1 * ... * m\tilde{k} <= n1 * ... * nk.
            The new distribution will output samples of shape (m1, ..., m\tilde{k}).

        Note that contrary to the general behavior for lin_transform for the Proba class,
        this method allows the ouput dimension to be strictly smaller than the input dimension, as
        long as the covariance matrix has trivial kernel (to avoid renormalisation issue for the
        log-density)
        """

        n_dim_shape = len(self._sample_shape)
        n_dim_sample = prod(self._sample_shape)

        mat = np.array(mat)
        mat_shape = mat.shape

        if mat_shape[-n_dim_shape:] != self._sample_shape:
            raise ShapeError(
                f"The shape of the matrix should end with {self._sample_shape}"
            )

        if prod(mat_shape[:-n_dim_shape]) > n_dim_sample:
            raise ShapeError(
                f"The first {n_dim_shape} dimensions of the matrix should multiply to less than {n_dim_sample}"
            )

        new_shape = mat_shape[:-n_dim_shape]

        mat = mat.reshape((n_dim_sample, n_dim_sample))

        means = (mat @ self._means).reshape(new_shape) + shift
        new_cov = (mat @ self._cov) @ mat.T

        return Gaussian(means=means, cov=new_cov, sample_shape=new_shape)

    def marginalize(self, indexes: list[int]):
        """
        Get the marginal distribution of (X_i)_{i in indexes} from distribution of X.

        Due to renormalization issues in the general case, this method is only possible
        for specific probability distributions such as Gaussian.

        The resulting distribution operates on 1D array.
        """
        new_mean, new_cov = self._means.copy(), self._cov.copy()  # type: ignore

        new_mean = new_mean[indexes]
        new_cov = new_cov[indexes]
        new_cov = new_cov[:, indexes]

        return Gaussian(new_mean, new_cov)
