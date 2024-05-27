r"""
Class for Exponential family of probability distributions.

Exponential families, using the natural parametrisation, have densities
..math::
    f_\theta(x) = \exp(\theta \cdot T(x) - g(\theta) + h(x))
with respect to a common distribution.

The Kullback--Leibler divergence has a closed form expression which amounts to a Bregman divergence
..math::
    KL(f_a, f_b) = g(b) - g(a) - (b - a) . nabla g(a).

This allows for easy differentiation, provided the Hessian of $g$ is known.

Reference:
    https://www.lix.polytechnique.fr/~nielsen/EntropyEF-ICIP2010.pdf

Note:
    Exponential family can be used to obtain another parametrizsation of Gaussian distributions as
        well as Gamma distributions. These specific implementations are expected to be somewhat
        more efficient though.
    Tensorization of Exponential families are also exponential families. While this information is
        lost, the map_tensorize function is coded in such a way as to ensure efficiency when
        computing kl, grad_kl, grad_right_kl methods.
"""

import warnings
from typing import Callable, Optional

import numpy as np
from surpbayes.misc import _get_pre_shape
from surpbayes.proba._errors import RenormError
from surpbayes.proba.proba import Proba
from surpbayes.proba.proba_map import ProbaMap
from surpbayes.proba.warnings import MissingShape, NegativeKLWarning
from surpbayes.types import ProbaParam, Samples


def normalised_check(g: Callable[[ProbaParam], float]) -> Callable[[ProbaParam], float]:
    r""" Decorator for log renormalisation functions, changing failure behavior (infinite,
    exception) to raising RenormError exception.
    
    Arg:
        g: the log renormalisation function
    return:
        fun: modified log renormalisation function, raising RenormError if g(proba_par) raises
            an exception or is infinite, and is not returns g(proba_par)
    """
    def fun(proba_param: ProbaParam):
        try:
            out = g(proba_param)
        except Exception as exc:
            raise RenormError(f"Renormalisation failed at {proba_param}") from exc
        if (out == np.inf) or (out == -np.inf) or (np.isnan(out)):
            raise RenormError(f"Renormalisation failed at {proba_param}")
        return out

    return fun


class ExponentialFamily(ProbaMap):
    r"""
    Subclass of ProbaMap for Exponential families.

    Exponential families have densities of form
    ..math::
        f_\†heta(x) = \exp(\theta \cdot T(x) - g(\theta) + h(x))

    (h can be omitted since it can be hidden in the reference measure).

    Many families of distributions are exponential families (gaussians, gamma, etc).
    """

    # Indicate that this is a ExponentialFamily object
    map_type = "ExponentialFamily"

    def __init__(  # pylint: disable=R0913
        self,
        gen: Callable[[ProbaParam], Callable[[int], Samples]],
        T: Callable[[Samples], np.ndarray],
        g: Callable[[ProbaParam], float],
        der_g: Callable[[ProbaParam], ProbaParam],
        der_der_g: Optional[Callable[[ProbaParam], np.ndarray]] = None,
        der_T: Optional[Callable[[Samples], np.ndarray]] = None,
        h: Optional[Callable[[Samples], np.ndarray]] = None,
        der_h: Optional[Callable[[Samples], np.ndarray]] = None,
        proba_param_shape: Optional[tuple] = None,
        sample_shape: Optional[tuple] = None,
        ref_param: Optional[ProbaParam] = None,
        np_out: Optional[bool] = None,
    ):
        r"""
        Proba map for an exponential family defined through its natural parameters

            $f_{\theta}(x) = \exp(\theta. T(x) - g(\theta) + h(x))$

        where f is the density.

        Natural parametrisation is required to efficiently compute KL. For change of
        parametrisation, use reparametrize which maintains efficient computation of KL
        and its gradient.
        """
        normed_log_dens = h is not None

        g = normalised_check(g)

        if (ref_param is None) & (proba_param_shape is None):
            warnings.warn(
                "No shape information on expected distribution parameters",
                category=MissingShape,
            )

        if proba_param_shape is None:
            proba_param_shape = np.array(ref_param).shape

        # Define dimensions on which to sum for log_dens function
        dims_log_dens_help = tuple(-i - 1 for i in range(len(proba_param_shape)))

        def prob_map(proba_param: ProbaParam) -> Proba:
            """Transforms a distribution parameter into a distribution (Proba object)"""
            loc_gen = gen(proba_param)

            g_loc = g(proba_param)
            if (g_loc == np.inf) or (g_loc == -np.inf) or (np.isnan(g_loc)):
                raise RenormError("Can not renormalise")

            if normed_log_dens:

                def log_dens(samples: Samples) -> np.ndarray:
                    # Samples should be of shape (pre_shape, sample_shape)
                    # T(samples) is of shape (pre_shape, proba_param_shape)
                    # h(samples) of shape (pre_shape,)
                    # g_loc should be a float

                    return (
                        (proba_param * T(samples)).sum(axis=dims_log_dens_help)
                        + h(samples)  # type: ignore
                        - g_loc
                    )

            else:

                def log_dens(samples: Samples) -> np.ndarray:
                    # Samples should be of shape (pre_shape, sample_shape)
                    # T(samples) is of shape (pre_shape, proba_param_shape)

                    return (proba_param * T(samples)).sum(
                        axis=dims_log_dens_help
                    ) - g_loc

            return Proba(
                gen=loc_gen, log_dens=log_dens, sample_shape=sample_shape, np_out=np_out
            )

        def log_dens_der(proba_param):
            g_der_loc = der_g(proba_param)

            def der(samples: Samples):
                return T(samples) - g_der_loc

            return der

        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            ref_param=ref_param,
            proba_param_shape=proba_param_shape,
            sample_shape=sample_shape,
        )

        self._normed_log_dens = normed_log_dens
        self._gen = gen
        self._g = g
        self._der_g = der_g
        self._H_g = der_der_g
        self._T = T
        self._der_T = der_T
        self._h = h
        self._der_h = der_h

    @property
    def t_shape(self):
        """Shape of ProbaParam (for compatibility with PreExpFamily)"""
        return self._proba_param_shape

    @property
    def normed_log_dens(self):
        return self._normed_log_dens

    @property
    def gen(self):
        """Generating function of the exponential family"""
        return self._gen

    @property
    def g(self):
        """Normalisation function."""
        return self._g

    @property
    def der_g(self):
        """Gradient of normalisation function."""
        return self._der_g

    @property
    def H_g(self):
        """Hessian of normalisation function (Optional).
        Raise ValueError if None."""
        if self.H_g is None:
            raise ValueError("No implementation for H_g")
        return self._H_g

    @property
    def T(self):
        r"""T function defining the exponential family (log densities
        of form :math:`\theta\cdot T - h(x) + g(\theta)`"""
        return self._T

    @property
    def der_T(self):
        """Derivative of T function"""
        return self._der_T

    @property
    def h(self):
        """Carrier measure (None interpreted as Lebesgue)"""
        return self._h

    def kl(
        self,
        param_1: ProbaParam,
        param_0: ProbaParam,
        n_sample: Optional[int] = None,
    ):
        """
        Compute the Kullback Leibler divergence between two distributions
        defined by their prior parameters.

        Args:
            param_1, param_0 are 2 proba parameters
            n_sample: disregarded

        Output:
            KL(proba_1, proba_0) computed through
                g(param_0) - g(param_1) - (param_0 - param_1) . nabla g(param_1)

        Reference:
            https://www.lix.polytechnique.fr/~nielsen/EntropyEF-ICIP2010.pdf
        """

        par1, par0 = np.asarray(param_1), np.asarray(param_0)
        if np.all(par1 == par0):
            return 0.0
        kl_out = (
            self.g(par0) - self.g(par1) - np.sum((par0 - par1) * self.der_g(param_1))
        )
        if kl_out < 0.0:
            warnings.warn(
                f"Found negative kl ({kl_out}). Returning 0", category=NegativeKLWarning
            )
            kl_out = 0.0
        return kl_out

    def grad_right_kl(
        self, param_1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        der_g1 = self.der_g(param_1)

        def der(param_0, n_sample: int = 0):  # pylint: disable=W0613
            return self.der_g(param_0) - der_g1, self.kl(param_1, param_0)

        return der

    def grad_kl(
        self, param_0: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        (nabla_{param_1} KL(param_1, param_0))

        Args:
            param_0 is a distribution parameter

        Output:
            If the hessian of the renormalisation is known, then this is used to compute the gradient.
            Else falls back to standard computations.

        Reference:
            Starting back from the formula for KLs of exponential families,
                KL(proba_1, proba_0) =
                    g(param_0) - g(param_1) - (param_0 - param_1) . nabla g(param_1)
            it follows that the gradient of the kl wrt to param_1 is
                Hessian(g)(param_1) (param_1 - param_0)
        """
        if self.H_g is None:
            return ProbaMap.grad_kl(self, param_0=param_0)

        indices = list(range(len(self.proba_param_shape)))

        def der(param_1: ProbaParam, n_sample: int = 0):  # pylint: disable=W0613
            par1, par0 = np.array(param_1), np.array(param_0)
            return np.tensordot(  # type: ignore
                self.H_g(param_1),  # type: ignore # (proba_param_shape, proba_param_shape)
                (par1 - par0),  # (proba_param_shape)
                [indices, indices],
            ), self.kl(param_1, param_0)

        return der

    def transform(
        self,
        transform: Callable[[Samples], Samples],
        inv_transform: Callable[[Samples], Samples],
        der_transform: Optional[Callable[[Samples], np.ndarray]] = None,
    ):
        r"""
        Transform the Class of probability $X_\theta \sim \mathbb{P}_{\theta}$ to the class of probability
            $transform(X_\theta)$

        Important:
            transform MUST be bijective, else computations for log_dens_der, kl, grad_kl, grad_right_kl will fail.

        CAUTION:
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

            Due to this, log_dens_der, kl, grad_kl, grad_right_kl will perform satisfactorily.

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
            Decide whether der_transform must output np.ndarray or not. Currently, does not have to (but could be less
            efficient since reforce array creation.)
        """
        # Some inferences
        try:
            new_sample_shape = np.array(self._map(self.ref_param)(1)).shape[1:]
        except Exception as exc:  # pylint: disable=W0703
            raise ValueError("Could not infer new sample_shape") from exc

        sample_size = self._sample_size

        def new_gen(proba_param: ProbaParam) -> Callable[[int], Samples]:
            fun = self.gen(proba_param)

            def new_fun(n: int) -> Samples:
                return transform(fun(n))  # type: ignore

            return new_fun

        def new_T(x: Samples) -> np.ndarray:
            return self.T(inv_transform(x))

        if der_transform is not None:

            def new_h(samples: Samples):
                pre_shape = _get_pre_shape(samples, new_sample_shape)

                ys = inv_transform(samples)
                ders = np.array(der_transform(ys)).reshape(  # type: ignore
                    pre_shape + (sample_size, sample_size)
                )

                return (self.h(ys) - np.log(np.abs(np.linalg.det(ders)))).reshape(  # type: ignore
                    pre_shape
                )

        else:
            # No longer density wrt to Lebesgue...
            new_h = self.h  # type: ignore

        return ExponentialFamily(
            gen=new_gen,
            T=new_T,
            g=self.g,
            der_g=self.der_g,
            der_der_g=self.H_g,
            der_T=None,
            h=new_h,
            der_h=None,  # Requires second order derivative of transform
            proba_param_shape=self.proba_param_shape,
            sample_shape=new_sample_shape,
            ref_param=self.ref_param,
            np_out=True,
        )
