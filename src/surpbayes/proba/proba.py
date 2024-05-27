"""
Proba class
a. Initialization
A Proba object describes a specific probability distribution. It is defined from 4 objects:
- A gen function, which takes as input an integer n, outputs n samples (shaped (n, sample_shape))
- A log_dens function (optional), which takes as input Samples and outputs the log density of the
distribution evaluated at these samples. The density is presumed to be computed wrt to Lebesgue (
this is not mandatory, but most transformative methods try to preserve this assumption).
- sample_shape, a tuple of integer specifying the shape of a single sample (not mandatory, is
inferred from a sample if not provided)
- np_out, a boolean specifying if the samples generated from gen are store in numpy.ndarray
objects. Inferred from a sample if not provided.

Note that the coherence between inputs is not assessed and is the user's responsability.

b. Attributes
- gen, see initialization
- log_dens, see initialization
- sample_shape, see initialization
- _sample_size, the size of a single sample (integer)

c. Methods
- __call__: uses gen.
- dens: short for np.exp(proba.log_dens(...))
- kl: estimate Kullback-Leibler divergence between distributions. proba.kl(proba_2, n) estimates
kl(proba, proba_2) using n samples generated from proba.
- f_div: approximate the f-divergence between distributions. proba.f_div(proba_2, f, n) estimates
D_f(proba, proba_2) using n samples generated from proba_2.
- integrate: estimate the expected value of a function using a sample of specified size.
- reshape: outputs a modified copy of the proba object, which outputs reshaped samples.
- flatten: outputs a modified copy of the proba object, which outputs flat samples.
- contract: from distribution on X, outputs the distribution on alpha * X.
- shift: from distribution on X, outputs the distributin on c + X.
- lin_transform: from distribution on X, ouputs the distribution on c + mat @ X.
- transform: from distribution on X, outputs the distribution on f(X) for bijective f (inverse
must be provided, Lebesgue reference measure is preserved if derivative of f is provided).

d. Functions manipulating proba objects
- tensorize: from a list of distributions of random variable X_1, ..., X_n, returns the
distribution of (X_1, ..., X_n), where each component is independent. See function documentation
for shape combination rules.
- mixture: from a list of distributions of random variables X_1, ..., X_n defined on same space
(i.e. sample_shape arguments are all equal), returns the mixture distribution (mixture weights can
be provided, by default same weights are used).
- from_sample: from a sample of points (x_1, ..., x_n) and a random variable X, returns the mixture
of X_1, ..., X_n where X_i ~ x_i + X.
- add: from distribution X and Y, returns the distribution of X+Y (independent sum). The
convolution in the log density is approximated using a sample of size specified at the construction
time. Not recommanded for use.


Implementation rules:

    The shape of the output after conversion to np.ndarray is known and can be assessed through
    'sample_shape' attribute. This property is NEVER None.

    The preferred class for the sample is np.ndarray. If so, the generator function should
    output np.ndarray rather than list of np.ndarray. The distribution object should work for
    other outputs nonetheless.

    In any case, the log_density function is expected to work after conversion to np.ndarray of the
    input.

Note on output class.
    User can define distribution on exotic object as long as they can be converted to np.ndarray of
    identical shape. If so, the output of gen can be any iterable.

    If np_out is specified as True, it is assumed that the output of gen is a np.ndarray of shape
        (n,) + sample_shape.

    Most methods which transform the probability distribution will output distributions such that
    np_out is True. These methods are: 'reshape', 'flatten', 'contract', 'shift', 'lin_transform'.
    The 'transform' method is a special case. If the output of transform is a np.ndarray, then the
    output distribution will be such that np_out is True. If the output is not a np.ndarray, then
    the output will be a list and np_out will be False.
"""

import warnings
from typing import Callable, Optional, Sequence, Union, overload

import numpy as np
from surpbayes.misc import (ShapeError, _get_pre_shape, par_eval, prod,
                            vectorize)
from surpbayes.proba.warnings import NegativeKLWarning, ShapeWarning
from surpbayes.types import SamplePoint, Samples


class Proba:
    """Probability distribution class

    Attributes:
        gen: a function of n, outputing a list-like object containing n i.i.d. sample from the
            distribution.
        log_dens: a function taking samples as input, outputing the log density of the
            distribution evaluated at those samples. This function should be vectorized.
        sample_shape: the shape of a single sample point
        _np_out: specify if gen outputs numpy.ndarray objects.

    Assumption:
        It is assumed that the distribution outputs np.ndarray like objects. Whether or not the
    samples are np.ndarray is assessed and has an impact on time performance for transforms.

    Remark:
        The log_dens function is defined up to a reference distribution. To avoid confusion,
    good practice consists in using the Lebesgue measure as reference distribution whenever
    possible. Note that the 'dens' and 'kl' methods are impacted by the choice of reference
    distribution. See documentation for these methods.

    <!> Warning <!>
        It is bad practice to change manually the values of the attributes _gen or _log_dens
        (hence they are considered as protected)
    Modifying these attributes will have side effects on other distributions constructed from this
    distribution (through reshape, flatten, lin_transform, transform).

        Risky:
        >>> proba = Proba(gen1, log_dens1, shap1)
        >>> proba_2 = proba.reshape(shape2)
        >>> proba._gen, proba._log_dens = gen2, log_dens2
        At this stage, proba_2 relies on the gen2 and log_dens2 functions to generate samples and
        compute log density.

        Rather:
        >>> proba = Proba(gen1, log_dens1)
        >>> proba_2 = proba.reshape(new_shape)
        >>> proba = Proba(gen2, log_dens2)
        At this stage, proba_2 relies on gen1 and log_dens1 functions to generate samples and
        compute log density.
    """

    def __init__(
        self,
        gen: Callable[[int], Samples],
        log_dens: Callable[[Samples], np.ndarray],
        sample_shape: Optional[tuple] = None,
        np_out: Optional[bool] = None,
    ):
        """
        Construct a Proba object from generator and log density functions.
        No checks are performed to assess that the log density function and Proba object are
        coherent.

        The density __should__ if possible be with respect to Lebesgue in order to preserve the
        reference measure under transformations.

        Arguments:
            gen: the generator function (input n:int, output a sample of size n).
            log_dens: the log_density function (input samples:Samples, output log density at
                samples)
            dimensional np.ndarray containing the log densities evaluated at the samples).
            sample_shape: the shape of the SamplePoint. Optional, inferred if not provided.
            np_out: Is the output a np.ndarray? Optional, inferred if not provided.
        """
        self._gen = gen
        self._log_dens = log_dens

        if (sample_shape is None) or (np_out is None):
            sample = gen(1)
            if sample_shape is None:
                sample_shape = np.array(sample[0]).shape
            if np_out is None:
                np_out = isinstance(sample, np.ndarray)

        self._sample_shape = sample_shape
        self._sample_size = prod(sample_shape)

        self._np_out = np_out

    @property
    def gen(self):
        """Function to sample from probability distribution"""
        return self._gen

    @property
    def log_dens(self):
        """Function to evaluate log density"""
        return self._log_dens

    @property
    def sample_shape(self):
        """Shape of generated samples.
        A new Proba object with updated sample_shape can be
        created through '.reshape' method.
        """
        return self._sample_shape

    @property
    def sample_size(self):
        """Size of generated samples"""
        return self._sample_size

    @property
    def np_out(self):
        """Whether samples are outputted as np.ndarray"""
        return self._np_out

    def __call__(self, n: int) -> Samples:
        """
        Generate an i.i.d. sample of length 'n'
        """
        return self._gen(n)

    def __repr__(self):
        return f"Probability distributions on ArrayLike of shape {self.sample_shape}"

    # Operations
    def dens(self, samples: Samples) -> np.ndarray:
        """
        Compute the density of the distribution at points xs.

        Args:
            samples, a list of samples where the density is to be evaluated.

        Remark:
            The density of the distribution depends on the reference distribution used
            when defining the log_dens function.
        """
        return np.exp(self._log_dens(samples))

    def kl(self, proba_2, n_sample: int = 1000) -> float:
        """
        Estimate the Kullback Leibler divergence with another probability (a.kl(b) = kl(a,b)).

        Args:
            proba_2: Proba object with respect to which the divergence is computed
            n_sample: number of samples used to approximate the divergence

        Remark:
            This KL computation is not exact. Implement closed form equations whenever
                possible instead of relying on this routine.
            It is possible that the KL computation gives a negative result. A NegativeKLWarning
                warning is raised whenever that happens.
            The routine relies on the log_dens attribute of the distributions.
                This attribute depends on a reference distribution. The KL method assumes that,
                and only works if, the reference distribution is the same for the two
                distributions.
        """
        if proba_2.sample_shape != self.sample_shape:
            raise ValueError(
                "Probability distributions are not defined on the same space"
            )
        samples = self._gen(n_sample)

        log_ratios = self._log_dens(samples) - proba_2._log_dens(samples)

        kl = np.mean(log_ratios)

        if kl < 0:
            warnings.warn(
                f"Negative KL ({kl}) - consider raising 'n_sample' parameter",
                category=NegativeKLWarning,
            )
        return kl

    def f_div(
        self,
        proba_2,
        f: Callable[[Sequence[float]], Sequence[float]],
        n_sample: int = 1000,
    ) -> np.floating:
        """
        Estimate the f divergence a.f_div(b,f) = D_f(a,b) using n samples.

        Remark:
            Function f must be convex and such that f(1) = 0 (no checks are performed).
            f is assumed to be vectorized (applies function component wise).
            It is possible that the KL computation gives a negative result. A NegativeKLWarning
        warning is raised whenever that happens.
            The routine relies on the log_dens attribute of the distributions. This attribute
        depends on a reference distribution. The KL method assumes that, and only works if, the
        reference distribution is the same for the two distributions.
        """
        samples = proba_2.gen(n_sample)

        log_ratios = self._log_dens(samples) - proba_2.log_dens(samples)

        evals = f(np.exp(log_ratios))

        f_div = np.mean(evals)

        if f_div < 0:
            warnings.warn(
                " ".join(
                    [
                        f"Negative F div ({f_div})",
                        "- Consider raising n_sample parameters or check f implementation",
                    ]
                ),
                category=NegativeKLWarning,
            )
        return f_div

    @overload
    def integrate(
        self,
        fun: Callable[[SamplePoint], float],
        n_sample: int,
        vectorized: bool,
        parallel: bool,
        **kwargs,
    ) -> float:
        ...

    @overload
    def integrate(
        self,
        fun: Callable[[SamplePoint], np.ndarray],
        n_sample: int,
        vectorized: bool,
        parallel: bool,
        **kwargs,
    ) -> np.ndarray:
        ...

    def integrate(
        self,
        fun: Union[Callable[[SamplePoint], float], Callable[[SamplePoint], np.ndarray]],
        n_sample: int = 100,
        vectorized: bool = False,
        parallel: bool = False,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        r"""
        Estimate the expected value :math:`\mathbb{E}[fun(x)]` using i.i.d. samples.

        Args:
            fun: a function of a sample outputing array like results. The function can take other
                keywords arguments
            n_sample: number of samples generated to estimate the integral
            vectorized: boolean, specify if the function is vectorized (i.e. accept multiple
                samples). Default is False. If True, parallel is disregarded.
            parallel: boolean, specify if function evaluations should be parallelized
        **kwargs are passed to func
        """

        samples = self._gen(n_sample)
        if vectorized:
            vals = fun(samples, **kwargs)
        else:
            vals = np.array(
                par_eval(fun=fun, xs=samples, parallel=parallel, **kwargs)
            )  # Note: array conversion is used only to assess shape.

        if len(vals.shape) > 1:  # type: ignore
            return np.apply_along_axis(np.mean, 0, vals)

        return np.mean(vals, dtype=float)

    # Transformations
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
        assume_np = self._np_out

        old_shape = self.sample_shape
        old_size = self._sample_size
        if prod(new_shape) != old_size:
            raise ShapeError("Can not reshape the distribution")

        if assume_np:

            def gen(n: int) -> Samples:
                return self._gen(n).reshape((n,) + new_shape)

        else:

            def gen(n: int) -> Samples:
                return np.array(self._gen(n)).reshape((n,) + new_shape)

        def log_dens(samples: Samples) -> np.ndarray:
            pre_shape = _get_pre_shape(samples, new_shape)
            return self._log_dens(samples.reshape(pre_shape + old_shape))

        return Proba(gen=gen, log_dens=log_dens, sample_shape=new_shape, np_out=True)

    def flatten(self):
        """
        Transform the distribution of X to the distribution of X.flatten()

        Note:
            The new distribution will generate np.ndarray objects
        """
        new_shape = (prod(self.sample_shape),)
        return self.reshape(new_shape)

    def contract(self, alpha: float):
        """
        Transform the distribution of X to the distribution of alpha * X

        Argument:
            alpha: a float

        Note:
            The new distribution will generate np.ndarray objects
        """

        if alpha == 0.0:
            raise ValueError("Contraction factor alpha must be non-0")

        assume_np = self._np_out
        log_det = self._sample_size * np.log(np.abs(alpha))

        if assume_np:

            def gen(n: int) -> Samples:
                return alpha * self._gen(n)

        else:

            def gen(n: int) -> Samples:
                return alpha * np.array(self._gen(n))

        def log_dens(samples: Samples) -> np.ndarray:
            return self._log_dens(samples / alpha) - log_det

        return Proba(
            gen=gen, log_dens=log_dens, sample_shape=self.sample_shape, np_out=True
        )

    def shift(self, shift: SamplePoint):
        """
        Transform the distribution of X to the distribution of X + shift

        Note:
            The new distribution will generate np.ndarray objects
        """
        if self._np_out:

            def gen(n: int) -> Samples:
                return (self._gen(n).T + shift).T

        else:

            def gen(n: int) -> Samples:
                return (np.array(self._gen(n)).T + shift).T

        def log_dens(samples: Samples) -> np.ndarray:
            return self._log_dens(samples - shift)

        return Proba(
            gen=gen, log_dens=log_dens, sample_shape=self.sample_shape, np_out=True
        )

    def lin_transform(self, mat: np.ndarray, shift: Union[float, SamplePoint] = 0.0):
        """
        Transform the distribution of X to the distribution of mat @ X + shift
        (where @ denote a full tensor product rather than matrix product).

         Shift can be either a float or a np.array object of shape compatible with the matrix.

        Dimension formating:
            If proba outputs samples of shape (n1, ..., nk), then the matrix should be shaped
                (m1, ..., m \tilde{k}, n1, ..., nk)
            with m1, ..., m \tilde k such that m1 * ... * m\tilde{k} = n1 * ... * nk.
            The new distribution will output samples of shape (m1, ..., m\tilde{k}).

        Future:
            Check that the current construction using reshape does not result in much time
            increasing during operations.
        """
        # Check proba object input/output
        assume_np = self._np_out

        n_dim_shape = len(self.sample_shape)
        n_dim_sample = self._sample_size
        old_shape = self.sample_shape

        # Force convert to array
        mat = np.array(mat)
        mat_shape = mat.shape

        # Assert shape coherence between matrix and sample
        if mat_shape[-n_dim_shape:] != old_shape:
            raise ShapeError(f"The shape of the matrix should end with {old_shape}")

        if len(mat_shape) <= n_dim_sample:
            raise ShapeError(
                f"Matrix should be shaped (new_shape, old_shape). Shape passed is {mat_shape}"
            )

        if prod(mat_shape[:-n_dim_shape]) != n_dim_sample:
            raise ShapeError(
                f"The first {n_dim_shape} dimensions of the matrix should multiply to {n_dim_sample}"
            )

        # Infer shape of new distribution
        new_shape = mat_shape[:-n_dim_shape]

        # Work on flat problem
        mat = mat.reshape((n_dim_sample, n_dim_sample))
        mat_t = mat.T

        # Pre compute quantities
        log_det = np.log(np.abs(np.linalg.det(mat)))
        inv_mat_t = np.linalg.inv(mat_t)

        # Prepare functions gen and log_dens for a probability with flat sample shape
        if assume_np:

            def gen(n: int):
                return self._gen(n) @ (mat_t) + shift

        else:

            def gen(n: int):
                return np.array(self._gen(n)) @ (mat_t) + shift

        def log_dens(samples: Samples) -> np.ndarray:
            # We assume that samples are flat since this corrected at a later stage

            # Get pre_shape
            pre_shape = samples.shape[:-1]

            # Convert to old samples
            old_xs = ((samples - shift) @ inv_mat_t).reshape(pre_shape + old_shape)

            # Correct with log_det
            return self._log_dens(old_xs) - log_det

        return Proba(
            gen=gen, log_dens=log_dens, sample_shape=(n_dim_sample,), np_out=True
        ).reshape(new_shape)

    def transform(
        self,
        transform: Callable[[Samples], Samples],
        inv_transform: Callable[[Samples], Samples],
        der_transform: Optional[Callable[[Samples], np.ndarray]] = None,
    ):
        """
        Transform the distribution of X to the distribution of transform(X)

        CAUTION:
            Implementation assumes that 'transform', 'inv_transform', 'der_transform' are
            vectorized.

            Everything possible is done to insure that the reference distribution remains
            Lebesgue IF the original reference distribution is Lebesgue. This requires access
            to the derivative of the transform (more precisely its determinant). If this can
            not be computed, then the log_density attribute will no longer be with reference to
            the standard Lebesgue measure.

            Still,
                proba_1.transform(f, inv_f).log_dens(x) - proba_2.transform(f, inv_f).log_dens(x)
            acccurately describes
                log (d proba_1 / d proba_2 (x)).

            If only ratios of density are to be investigated, der_transform can be disregarded.
            In that case, it makes more sense to consider the ratio of density on the original space
            directly, though.
        CAUTION:
            If transform is not one on one, even if transform o inv_transform = Id, the log_dens function
            of proba.transform(transform, inv_transform) will NOT be usable in the general case.

        Dimension formating:
            If proba outputs samples of shape (s1, ..., sk), and transforms maps them to (t1, ..., tl)
             then the derivative should be shaped
                (s1, ..., sk, t1, ..., tl).
            The new distribution will output samples of shape (t1, ..., tl).

            Moreover, transform, inv_transform and der_transform are assumed to be vectorized, i.e.
            inputs of shape (n1, ..., np, s1, ..., sk) will result in outputs of shape
                (n1, ..., np, t1, ..., tl ) for transform, (n1, ..., np, s1, ..., sk, t1, ..., tl ) for
                der_transform

        Output formating:
            The format of the output depends on the behavior of transform. If transform outputs np.ndarray,
            then the resulting distribution will have np_out==True.

        Future:
            Decide whether der_transform must output np.ndarray or not. Currently, does not have to
                (but could be less efficient since reforce array creation.)
        """
        sample_size = self._sample_size

        def gen(n: int) -> Samples:
            return transform(self._gen(n))

        new_sample_shape = np.array(gen(1)).shape[1:]

        if der_transform is None:

            def log_dens(xs: Samples) -> np.ndarray:
                return self._log_dens(inv_transform(xs))

        else:

            def log_dens(xs: Samples) -> np.ndarray:
                pre_shape = _get_pre_shape(xs, new_sample_shape)

                ys = inv_transform(xs)  # shape (pre_shape, old_sample_shape)
                ders = np.array(der_transform(ys)).reshape(  # type: ignore
                    pre_shape + (sample_size, sample_size)
                )

                return (
                    self._log_dens(ys) - np.log(np.abs(np.linalg.det(ders)))
                ).reshape(pre_shape)

        return Proba(gen=gen, log_dens=log_dens, sample_shape=new_sample_shape)


def tensorize(*probas: Proba, flatten=True, dim: int = 0) -> Proba:
    r"""
    From a collection of distribution :mathbb:`mu_i`, outputs distribution :math:`(mu_1, mu_2, \dots)`
    (with independent components)

    Shape issues:
        By default, the distributions are flattened, i.e. the routine considers their outputs as
        independant 1 dimensional arrays, and the output of tensorize is a distribution on
        1 dimensional array.

        It is possible to construct distributions on multi dimensional arrays if the shape of both
        distributions are compatible.
        If the first distribution is a distribution on arrays of shape
            (d1, d2, ...,     d dim 0,    ..., d k),
        and all other distributions are a distribution on arrays of shape
            (d1, d2, ...,     d dim i,    ..., d k),
        then Proba.tensorize(proba_0, proba_1, ..., flatten=False, dim=dim) is the tensorized
        distribution on arrays of shape
            (d1, d2, ...,  sum d dim i  , ..., d_k)

        If the shapes are not compatible, then tensorize reverts to the flattening behavior.

    <!> USER WARNING <!>
    The output of tensorize is linked to the distributions from which it was generated.
    For instance, if proba = tensorize(d1, d2), if attributes of d1 or d2 are modified, proba might
    change.
    """

    # Do not go through all that hokus pokus if only a single distribution is passed
    if len(probas) == 1:
        return probas[0]

    shapes = [proba.sample_shape for proba in probas]

    # Check whether not flattened tensorization is possible
    if not flatten:
        # Check that all the shapes are compatible
        pre_shapes = [shape[:dim] for shape in shapes]
        post_shapes = [shape[(dim + 1) :] for shape in shapes]

        ok_pre = pre_shapes.count(pre_shapes[0]) == len(pre_shapes)
        ok_post = post_shapes.count(post_shapes[0]) == len(post_shapes)

        if ok_pre and ok_post:
            pre_shape = pre_shapes[0]
            post_shape = post_shapes[0]

            d_dim = [shape[dim] for shape in shapes]
            tot_dim = sum(d_dim)
            sample_shape = pre_shape + (tot_dim,) + post_shape

        else:
            warnings.warn(
                "\n".join(
                    [
                        "Shapes of distributions do not match",
                        "Distributions will be flattened before tensorizing.",
                    ]
                ),
                category=ShapeWarning,
            )
            flatten = True

    # Main function call if flatten = False
    if not flatten:
        pivot = np.cumsum([0] + d_dim)

        def gen(n: int) -> Samples:
            samples = [proba._gen(n) for proba in probas]

            return np.concatenate(
                samples, axis=dim + 1
            )  # +1 since the first dimension is due to the number of samples used

        def log_dens(samples: Samples) -> np.ndarray:
            samples = np.array(
                samples
            )  # Array of shape (pre_shape, d0, ...sum d_dim_i, ... dmax)
            pre_shape = _get_pre_shape(samples, sample_shape)
            samples = samples.reshape((prod(pre_shape),) + sample_shape)

            xs_tilde = np.moveaxis(
                samples, dim + 1, 0
            )  # Array of shape (sum d_dim, d0, ...n, ... dmax)

            xs_s = [
                np.moveaxis(
                    xs_tilde[p0:p1], 0, dim + 1
                )  # Back to shape n, d0, ..., d_dim_i, ..., dmax)
                for p0, p1 in zip(pivot[:-1], pivot[1:])
            ]

            log_denses = np.array(
                [proba._log_dens(x_s) for proba, x_s in zip(probas, xs_s)]
            )

            # This is an array of shape (n_proba, n_samples)
            return log_denses.sum(0).reshape(pre_shape)

    else:
        dims = [prod(shape) for shape in shapes]
        sample_shape = (sum(dims),)

        pivot = np.cumsum([0] + dims)

        def gen(n: int) -> Samples:
            accu = np.zeros((n,) + sample_shape)
            for p0, p1, proba in zip(pivot[:-1], pivot[1:], probas):
                accu[:, p0:p1] = np.array(proba._gen(n)).reshape((n, p1 - p0))

            return accu

        def log_dens(samples: Samples) -> np.ndarray:
            pre_shape = _get_pre_shape(samples, sample_shape)

            n = prod(pre_shape)
            samples = samples.reshape(((n,) + sample_shape))

            return np.array(
                [
                    proba._log_dens(samples[:, p0:p1].reshape((n,) + shape))
                    for proba, p0, p1, shape in zip(
                        probas, pivot[:-1], pivot[1:], shapes
                    )
                ]
            ).sum(0)

    return Proba(gen=gen, log_dens=log_dens, sample_shape=sample_shape)


def mixture(*args: Proba, weights: Optional[Sequence[float]] = None) -> Proba:
    r"""
    From a tuple of Proba objects and optional weights, returns the mixture
    probability whose density satisfies
    ..math::
        \frac{d\mu}{d\pi} = \frac{\sum_i \omega_i \frac{d\mu_i}{d\pi}}{\sum_i \omega_i}

    <!> USER WARNING <!>
    The densities of all Proba objects should be computed with respect to the same
    measure. It this is not the case, the Proba object returned will be meaningless
    """

    # Checks on shape
    sample_shape = args[0].sample_shape
    n_dim_shape = len(sample_shape)
    for prob in args:
        if not sample_shape == prob.sample_shape:
            raise ShapeError(
                "All probabilities should generate samples with identical shapes."
            )

    n_args = len(args)
    if weights is None:
        nd_weights = np.ones(n_args) / n_args
    else:
        nd_weights = np.array(weights)
        nd_weights = nd_weights / np.sum(nd_weights)

    def log_dens(samples: Samples) -> np.ndarray:
        samples = np.array(samples)
        pre_shape = samples.shape[:-n_dim_shape]

        n_tot = prod(pre_shape)

        log_densities = np.array([d._log_dens(samples) for d in args]).reshape(
            (n_args, n_tot)
        ).T + np.log(
            nd_weights
        )  # Of shape (n_tot, n_args)

        # Remove 0 densities
        index_0s = (log_densities == -np.inf).all(-1)  # shape (n_tot,)

        log_densities = log_densities[~index_0s]
        max_densities = log_densities.max(-1)
        log_densities = (log_densities.T - max_densities).T
        log_dens_red = max_densities + np.log(np.exp(log_densities).sum(-1))

        tot = np.zeros(n_tot)
        tot[index_0s] = -np.inf
        tot[~index_0s] = log_dens_red
        return tot.reshape(pre_shape)

    def gen(n: int) -> Samples:
        ks = np.random.choice(len(args), n, p=nd_weights)
        return np.array([args[k]._gen(1)[0] for k in ks])

    return Proba(gen=gen, log_dens=log_dens)


def from_sample(
    sample: Sequence[SamplePoint], kernel: Proba, weights: Optional[list] = None
) -> Proba:
    r"""
    From a (weighted) - sample (:math:`X_i`, :math:`\omega_i`) and a kernel (distribution
    of random variable :math:`\epsilon`), returns the distribution of
    ..math::
        \epsilon + \sum \omega_i \delta_{X_i}

    where :math:`\delta_X` is the dirac distribution on X.
    """
    probas = tuple(kernel.shift(x) for x in sample)
    return mixture(*probas, weights=weights)


def add(prob1: Proba, prob2: Proba, n_sample=1000, parallel=False) -> Proba:
    r"""
    Independant addition of random variables.

    If :math:`X_1\sim prob1`, :math:`X_2 \sim prob2`, and :math:`X_1, X_2` are independent,
    then :math:`X_1 + X_2 \sim add(prob1, prob2)`

    This requires an approximation when computing the log_density function of the resulting
    distribution, since the formula involves an integral. It is moreover only usable only for
    distributions whose density is computed with respect to Lebesgue measure.
    """
    if prob1.sample_shape != prob2.sample_shape:
        raise ShapeError(
            f"Non conforming shape {prob1.sample_shape}, {prob2.sample_shape}"
        )

    def gen(n):
        return np.asarray(prob1._gen(n)) + np.asarray(prob2._gen(n))

    def single_log_dens(x: SamplePoint) -> float:
        samples = np.asarray(prob1._gen(n_sample))
        to_find = x - samples

        log_densities = prob2._log_dens(to_find)
        max_log_dens = np.max(log_densities)
        densities = np.exp(log_densities - max_log_dens)
        dens = np.mean(densities)
        log_dens = np.log(dens) + max_log_dens
        return log_dens

    log_dens = vectorize(
        single_log_dens,
        input_shape=prob1.sample_shape,
        convert_input=(not prob2._np_out),  # pylint: disable=W0212
        parallel=parallel,
    )

    return Proba(gen=gen, log_dens=log_dens, sample_shape=prob1.sample_shape)
