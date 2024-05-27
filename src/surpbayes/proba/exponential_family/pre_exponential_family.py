r"""
Class for Exponential families using a different parametrisation than the standard one

This class is designed with bayes module in view. It is basically a standard ProbaMap with
two more (functional) attributes:
- T
- T_to_param

It is assumed that the probability distributions have log_density of form :math:`T(x) \cdot F(param)`.
T_to_param is the function :math:`F^{-1}`, and param_to_T the function F.

In the case where the probability map is not injective, T_to_param should map to any parameter outputing
the distribution.

Used while no better solution is found.
"""
import warnings
from typing import Callable, Optional, Sequence

import numpy as np
from surpbayes.proba.proba import Proba
from surpbayes.proba.proba_map import ProbaMap
from surpbayes.proba.warnings import NegativeKLWarning
from surpbayes.types import ProbaParam, Samples


class PreExpFamily(ProbaMap):
    """Class for Exponential families using a different parametrisation than the natural on

    This class is designed with bayes module in view. It is basically a standard ProbaMap with
    three more (functional) attributes:
    - T (Samples to array)
    - T_to_param
    - param_to_T
    More Optional functional attributes can also be passed when constructing
    - g: the normalisation function (input in natural parametrisation)
    - der_g: the gradient of the normalisation function (input in natural parametrisation)
    - der_der_g: the Hessian of the normalisation function (input in natural parametrisation)

    It is assumed that the probability distributions have log_density of form T(x) . F(param) .
    T_to_param is the function :math:`F^{-1}`.

    In the case where the probability map is not injective, T_to_param should map to any parameter
    outputing the distribution.
    """
    # Indicate that this is a ExponentialFamily object
    map_type = "PreExpFamily"

    def __init__(
        self,
        prob_map: Callable[[ProbaParam], Proba],
        log_dens_der: Callable[[ProbaParam], Callable[[Samples], np.ndarray]],
        T: Callable[[Samples], np.ndarray],
        param_to_T: Callable[[ProbaParam], np.ndarray],
        T_to_param: Callable[[np.ndarray], ProbaParam],
        der_T_to_param: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        der_g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        der_der_g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        ref_param: Optional[ProbaParam] = None,
        proba_param_shape: Optional[tuple[int, ...]] = None,
        sample_shape: Optional[tuple[int, ...]] = None,
        t_shape: Optional[tuple[int, ...]] = None,
    ):
        super().__init__(
            prob_map, log_dens_der, ref_param, proba_param_shape, sample_shape
        )
        self._T = T
        self._param_to_T = param_to_T
        self._T_to_param = T_to_param
        self._der_T_to_param = der_T_to_param

        self._g = g
        self._der_g = der_g
        self._der_der_g = der_der_g

        if t_shape is None:
            if ref_param is not None:
                t_shape = prob_map(ref_param).sample_shape
            else:  # pylint: disable= W0702
                raise ValueError("Could not infer t_shape")
        self._t_shape = t_shape

    @property
    def t_shape(self):
        """Shape of expectation parametrisation (i.e. shape of natural parameter)"""
        return self._t_shape

    @property
    def g(self):
        """Log normalizer function (Optional). Takes natural parametrisation as input"""
        return self._g

    @property
    def der_g(self):
        """Gradient of log normalizer function (Optional). Takes natural parametrisation as input"""
        return self._der_g

    @property
    def T(self):
        """Sufficient statistic function"""
        return self._T

    @property
    def param_to_T(self):
        """Map from current parametrisation to natural parametrisation"""
        return self._param_to_T

    @property
    def T_to_param(self):
        """Map from natural parametrisation to current parametrisation"""
        return self._T_to_param

    @property
    def der_T_to_param(self):
        """Derivative of T_to_param function"""
        return self._der_T_to_param

    def kl(
        self,
        param_1: ProbaParam,
        param_0: ProbaParam,
        n_sample: int = 1000,):

        # Check if exponential family kl expression can be used
        if (self.g is None) or (self.der_g is None):
            return super().kl(param_1, param_0, n_sample)

        # Use closed form expression by default
        
        ppar1, ppar0 = np.asarray(param_1), np.asarray(param_0)

        #  Check if params are identical
        if np.all(ppar1 == ppar0):
            return 0.0

        # Convert to natural parametrisation
        par1, par0 = self.param_to_T(ppar1), self.param_to_T(ppar0)
        if np.all(par1 == par0):
            return 0.0

        # Compute KL
        kl_out = (
            self.g(par0) - self.g(par1) - np.sum((par0 - par1) * self.der_g(param_1))
        )
        if kl_out < 0.0:
            warnings.warn(
                f"Found negative kl ({kl_out}). Returning 0.0", category=NegativeKLWarning
            )
            kl_out = 0.0
        return kl_out

    def transform(
        self,
        transform: Callable[[Samples], Samples],
        inv_transform: Callable[[Samples], Samples],
        der_transform: Optional[Callable[[Samples], np.ndarray]] = None,
        new_sample_shape: Optional[tuple[int, ...]] = None,
    ):
        r"""
        Transform the Class of probability :math:`X_\theta \sim \mathbb{P}_{\theta}` to the class of
        probability :math:`transform(X_\theta)` for bijective "transform" map.

        Important:
            transform MUST be bijective, else computations for log_dens_der, kl, grad_kl,
            grad_right_kl will fail.


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

        FUTURE:
            Decide whether der_transform must output np.ndarray or not. Currently, does not have to (but could be less
            efficient since reforce array creation.)
        """

        def new_map(x: ProbaParam) -> Proba:
            return self._map(x).transform(transform, inv_transform, der_transform)

        if new_sample_shape is None:
            if self.ref_param is None:
                raise ValueError("Could not infer new sample shape")
            new_sample_shape = new_map(self.ref_param).sample_shape

        def new_log_dens_der(
            x: ProbaParam,
        ) -> Callable[[Samples], np.ndarray]:
            log_dens_der_fun = self._log_dens_der(x)

            def new_func(samples: Samples) -> np.ndarray:
                return log_dens_der_fun(inv_transform(samples))

            return new_func

        def new_T(samples: Samples):
            return self.T(inv_transform(samples))

        return TransformedPreExpFamily(
            prob_map=new_map,
            log_dens_der=new_log_dens_der,
            T=new_T,
            param_to_T=self.param_to_T,
            T_to_param=self.T_to_param,
            der_T_to_param=self.der_T_to_param,
            g=self.g,
            der_g=self.der_g,
            ref_param=self.ref_param,
            proba_param_shape=self.proba_param_shape,
            sample_shape=new_sample_shape,
            t_shape=self.t_shape,
            kl=self.kl,
            grad_kl=self.grad_kl,
            grad_right_kl=self.grad_right_kl,
            f_div=self.f_div,
            grad_f_div=self.grad_f_div,
            grad_right_f_div=self.grad_right_f_div,
        )


class TransformedPreExpFamily(PreExpFamily):
    """
    Class for transformed PreExpFamily. Reimplementations of kl and its derivatives are preserved
    
    Incentive:
        While the ExponentialFamily class uses the natural definition of the KL divergence using
        the gradient of the normalisation function, since the PreExpFamily uses a different
        parametrisation and is allowed not to have a normalisation function, it defaults to the
        standard approximation form of the KL. Such form is meant to be overridden by
        reimplementation of the method after inheritance; such is the case for the GaussianMap
        class, which uses a closed form expression in the mean/covariance parametrisation.
        Hence the necessity to protect both the PreExpFamily properties + the reimplemented
        divergences.
    """

    def __init__(
        self,
        prob_map: Callable[[ProbaParam], Proba],
        log_dens_der: Callable[[ProbaParam], Callable[[Samples], np.ndarray]],
        T: Callable[[Samples], np.ndarray],
        param_to_T: Callable[[ProbaParam], np.ndarray],
        T_to_param: Callable[[np.ndarray], ProbaParam],
        der_T_to_param: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        der_g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        ref_param: Optional[ProbaParam] = None,
        proba_param_shape: Optional[tuple[int, ...]] = None,
        sample_shape: Optional[tuple[int, ...]] = None,
        t_shape: Optional[tuple[int, ...]] = None,
        kl=None,
        grad_kl=None,
        grad_right_kl=None,
        f_div=None,
        grad_f_div=None,
        grad_right_f_div=None,
    ):
        super().__init__(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            T=T,
            param_to_T=param_to_T,
            T_to_param=T_to_param,
            der_T_to_param=der_T_to_param,
            g=g,
            der_g=der_g,
            ref_param=ref_param,
            proba_param_shape=proba_param_shape,
            sample_shape=sample_shape,
            t_shape=t_shape,
        )

        self._kl = kl
        self._grad_kl = grad_kl
        self._grad_right_kl = grad_right_kl

        self._f_div = f_div
        self._grad_f_div = grad_f_div
        self._grad_right_f_div = grad_right_f_div

    def transform(
        self,
        transform: Callable[[Samples], Samples],
        inv_transform: Callable[[Samples], Samples],
        der_transform: Optional[Callable[[Samples], np.ndarray]] = None,
        new_sample_shape: Optional[tuple[int, ...]] = None,
    ):
        r"""
        Transform the Class of probability :math:`X_\theta \sim \mathbb{P}_{\theta}` to the class of probability
            :math:`transform(X_\theta)` for bijective "transform".

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

        Future:
            Decide whether der_transform must output np.ndarray or not. Currently, does not have to (but could be less
            efficient since reforce array creation.)
        """

        def new_map(x: ProbaParam) -> Proba:
            return self._map(x).transform(transform, inv_transform, der_transform)

        if new_sample_shape is None:
            if self.ref_param is None:
                raise ValueError("Could not infer new sample shape")
            new_sample_shape = new_map(self.ref_param).sample_shape

        def new_log_dens_der(
            x: ProbaParam,
        ) -> Callable[[Samples], np.ndarray]:
            log_dens_der_fun = self._log_dens_der(x)

            def new_func(samples: Samples) -> np.ndarray:
                return log_dens_der_fun(inv_transform(samples))

            return new_func

        def new_T(samples: Samples):
            return self.T(inv_transform(samples))

        return TransformedPreExpFamily(
            prob_map=new_map,
            log_dens_der=new_log_dens_der,
            T=new_T,
            param_to_T=self.param_to_T,
            T_to_param=self.T_to_param,
            der_T_to_param=self.der_T_to_param,
            g=self.g,
            der_g=self.der_g,
            ref_param=self.ref_param,
            proba_param_shape=self.proba_param_shape,
            sample_shape=new_sample_shape,
            t_shape=self.t_shape,
            kl=self._kl,
            grad_kl=self._grad_kl,
            grad_right_kl=self._grad_right_kl,
            f_div=self._f_div,
            grad_f_div=self._grad_f_div,
            grad_right_f_div=self._grad_right_f_div,
        )

    def kl(  # pylint: disable=E0202
        self,
        param_1: ProbaParam,
        param_0: ProbaParam,
        n_sample: int = 1000,
    ) -> float:
        """Approximate the Kullback Leibler divergence between two distributions
        defined by their prior parameters.

        Args:
            param_1, param_0 are 2 prior parameters
            n_sample specifies how many points are used to estimate Kullback

        Output:
            kl(param_1, param_0) approximated as Sum_i(log(proba_1(phi_i)/proba_0(phi_i))
            with phi_i sampled through proba_1.gen (typically i.i.d.)

        Note:
            For a ProbaMap object obtained as the result of .reparametrize method with
            inherit_method=True, this method might be hidden behind a function attribute which
            is more efficient.
        """
        return self._kl(
            param_1,
            param_0,
            n_sample,
        )

    def grad_kl(  # pylint: disable=E0202
        self, param_0: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Approximates the gradient of the Kullback Leibler divergence between two distributions
        defined by their distribution parameters, with respect to the first distribution
        (nabla_{proba_1} kl(proba_1, proba_0))

        Args:
            param_0, a distribution parameter

        Output:
            A closure taking as arguments:
                param_1, a distribution parameter
                n_sample, an integer
            outputing a tuple with first element
                nabla_{param_1}kl(param_1, param_0) approximated using a sample
                    phi_i of predictors generated through proba_1.gen (typically i.i.d.).
                kl(param_1, param_0) approximated using the same sample of predictors

        This method should be rewritten for families with closed form expressions of KL and
        KL gradient. The closure form is used to simplify caching computations related to param_0
        (for instance inverse of covariance matrix for gaussian distributions).

        Note:
            For a ProbaMap object obtained as the result of .reparametrize method with
            inherit_method=True, this method might be hidden behind a function attribute which
            is more efficient.
        """

        return self._grad_kl(param_0)

    def grad_right_kl(  # pylint: disable=E0202
        self, param_1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Compute the derivative of the Kullback--Leibler divergence with respect to the second
        distribution.
        """
        return self._grad_right_kl(param_1)

    def f_div(  # pylint: disable=E0202
        self,
        param_1: ProbaParam,
        param_0: ProbaParam,
        f: Callable[[Sequence[float]], Sequence[float]],
        n_sample: int = 1000,
    ) -> float:
        r"""Approximates the f-divergence between two distributions
        defined by their prior parameters.

        Args:
            param_1, param_0 are 2 prior parameters
            f, a convex function such that f(1) = 0 (No checks are performed).
            n_sample, number of points used to estimate the f-divergence

        Output:
            :math:`D_f(proba_1, proba_0)` approximated as
            :math:`\sum_i(f(\frac{ d proba_1(\phi_i)}{d proba_0(\phi_i)})`
            with :math:`\phi_i` sampled through proba_0.gen (typically i.i.d.)

        Note:
            For a ProbaMap object obtained as the result of .reparametrize method with
            inherit_method=True, this method might be hidden behind a function attribute which
            is more efficient.
        """
        return self._f_div(param_1, param_0, f, n_sample)

    def grad_f_div(  # pylint: disable=E0202
        self,
        param_0: ProbaParam,
        f: Callable[[float], float],
        f_der: Callable[[float], float],
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        r"""Approximates the gradient of the f-divergence between two distributions
        defined by their prior parameters, with respect to the first distribution.

        Args:
            param_0, the parameter describing the second distribution.
            f, a convex function such that f(1) = 0 (No checks are performed).
                Should be vectorized
            f_der, the derivative of f. Should be vectorized
        """
        return self._grad_f_div(param_0, f, f_der)

    def grad_right_f_div(  # pylint: disable=E0202
        self,
        param_1: ProbaParam,
        f: Callable[[float], float],
        f_der: Callable[[float], float],
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        r"""Approximates the gradient of the f-divergence between two distributions
        defined by their prior parameters, with respect to the second distribution.

        Args:
            param_1, the parameter describing the first distribution.
            f, a convex function such that f(1) = 0 (No checks are performed).
            f_der, the derivative of f
        """
        return self._grad_right_f_div(param_1, f, f_der)
