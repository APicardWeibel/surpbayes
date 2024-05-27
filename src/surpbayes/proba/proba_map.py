r"""
Probability Map class.

For a parametric probability class, ProbaMap encodes the transform
..math::
    \alpha \rightarrow \mathbb{P}_\alpha.

This encoding also requires information on the derivative of the log density with respect to the
:math:`\alpha` parameter.
"""

import warnings
from typing import Callable, Optional, Sequence, Tuple, Union, overload

import numpy as np
from surpbayes.misc import (_get_pre_shape, blab, interpretation, par_eval,
                            post_modif, prod)
from surpbayes.proba.proba import Proba, tensorize
from surpbayes.proba.warnings import NegativeKLWarning
from surpbayes.types import ProbaParam, ProbaParams, SamplePoint, Samples


class ProbaMap:  # pylint: disable=R0902
    r"""
    ProbaMap class.
    Mapping for parametric family of distribution

    Attributes:
        map, the actual mapping
        log_dens_der, a function of PriorParam outputing a function of PredictParam
            outputing a PriorParam such
        :math:`\log_dens(x+ alpha, y ) - log_dens(x, y) \simeq log_dens_der(x)(y) \cdot \alpha`
        for all small prior parameters alpha.
        proba_param_shape, the shape of the parameter describing the distribution
        sample_shape, the shared shape of the sample of every distribution in the family

    Note on log_dens_der input/output:
    Assuming that sample\_shape = (s1, ..., sk) and that proba_param_shape is of shape (d1, ..., dp)
    Then log_dens_der(param) is a function which takes input of shape (n1, ..., nm, s1, ..., sk)
        and ouputs an array of shape (n1, ..., nm, d1, ..., dp)

    Future:
        Special case of integrate_der if the function is a Multivariate polynomial. Could be
        useful for gaussian distributions where the integration can be performed through exact
        computation.
    """

    # Indicate that this is a standard ProbaMap object
    map_type = "ProbaMap"

    def __init__(  # pylint: disable=R0913
        self,
        prob_map: Callable[[ProbaParam], Proba],
        log_dens_der: Callable[[ProbaParam], Callable[[Samples], np.ndarray]],
        ref_param: Optional[ProbaParam] = None,
        proba_param_shape: Optional[tuple[int, ...]] = None,
        sample_shape: Optional[tuple[int, ...]] = None,
    ):
        r"""
        Initialize a ProbaMap object.

        Args:
            prob_map: function which takes as input a ProbaParam (an array like) and outputs a
                Proba object.
            log_dens_der: function which takes as input a ProbaParam and outputs a closure. This
                closure takes as input Samples and returns the derivative of log densities of the
                distribution mapped by the ProbaParam, the derivative being with respect to the
                ProbaParam. Mathematically, it is the derivative of the function
                    ..math::
                        theta, x \rightarrow prob_map(theta).log_dens(x)
                with respect to :math:`\theta`.
            ref_param: reference ProbaParam. Optional.
            proba_param_shape: shape of ProbaParam objects accepted by prob_map. Optional.
            sample_shape: shared sample shape of the probability distributions outputed by
                prob_map. Optional

        """
        self._map = prob_map
        self._log_dens_der = log_dens_der

        if ref_param is not None:
            ref_param = np.array(ref_param)

        if proba_param_shape is None:
            if ref_param is None:
                raise ValueError(
                    " ".join(
                        [
                            "Could not infer shape of expected ProbaParam",
                            "('ref_param' and 'proba_param_shape' missing)",
                        ]
                    )
                )
            proba_param_shape = ref_param.shape
        if ref_param is None:
            # Try 0 parameter (can fail)
            warnings.warn(
                "\n".join(
                    [
                        "No reference parameter passed",
                        "Setting array full of 0 as reference parameter",
                    ]
                )
            )
            ref_param = np.zeros(proba_param_shape)
            try:
                # Try constructing a distribution with 0 param
                proba = prob_map(ref_param)
                # Check if that distribution can generate samples
                proba(1)
            except : # pylint: disable=W0702
                warnings.warn("Could not infer ref_param")

        self._proba_param_shape = proba_param_shape

        self._ref_param: ProbaParam = ref_param

        if sample_shape is None:
            sample_shape = self._map(ref_param)._sample_shape

        self._sample_shape = sample_shape

        self._sample_size: int = prod(sample_shape)

    @property
    def log_dens_der(self):
        return self._log_dens_der

    @property
    def sample_shape(self):
        return self._sample_shape

    @property
    def sample_size(self):
        return self._sample_size

    @property
    def ref_param(self):
        """Reference proba param"""
        return self._ref_param

    @ref_param.setter
    def ref_param(self, value):
        new_ref_param = np.array(value)
        if new_ref_param.shape != self.proba_param_shape:
            raise ValueError(
                f"New ref_param shape is inappropriate (expected {self.proba_param_shape})"
            )
        try:
            proba = self(new_ref_param)
        except Exception as exc:
            raise ValueError(
                f"New ref_param could not be interpreted by the map (param:{new_ref_param})"
            ) from exc
        if proba._sample_shape != self._sample_shape:
            raise ValueError("New ref_param does not output samples of adequate shape")

        self._ref_param = new_ref_param

    @property
    def proba_param_shape(self):
        return self._proba_param_shape

    def __call__(self, x: ProbaParam) -> Proba:
        """Short cut to the proba_map attribute"""
        return self._map(x)

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
        return self._map(param_1).kl(self._map(param_0), n_sample=n_sample)

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

        proba_0 = self._map(param_0)
        log_dens0 = proba_0._log_dens

        def fun(param_1: ProbaParam, n_sample: int) -> tuple[ProbaParam, float]:
            proba_1 = self._map(param_0)
            pred_sample = proba_1._gen(n_sample)

            ev_log1 = proba_1._log_dens(pred_sample)
            ev_log0 = log_dens0(pred_sample)
            eval_log = ev_log1 - ev_log0

            kl = np.mean(eval_log)
            eval_log = eval_log - kl  # shape n,

            log_der = self._log_dens_der(param_1)

            eval_der = log_der(pred_sample)  # shape n, proba_param_shape
            kl_der = (
                np.tensordot(  # type: ignore
                    eval_der, eval_log, (0, 0)  # (n, proba_param_shape)  # (n, )
                )
                / n_sample
            )  # (proba_param_shape)

            if kl < 0:
                warnings.warn(
                    f"Negative kl ({kl}) - consider raising n_sample parameter",
                    category=NegativeKLWarning,
                )

            return kl_der, kl

        return fun

    def grad_right_kl(  # pylint: disable=E0202
        self, param_1: ProbaParam
    ) -> Callable[[ProbaParam, int], tuple[ProbaParam, float]]:
        """
        Compute the derivative of the Kullback--Leibler divergence with respect to the second
        distribution.

        """
        proba_1 = self._map(param_1)

        def fun(param_0: ProbaParam, n_sample: int) -> tuple[ProbaParam, float]:
            # Derivative of KL evaluation
            sample = proba_1(n_sample)
            der_kl = -self._log_dens_der(param_0)(sample).mean(0)

            # Evaluation of KL
            ev_log1 = proba_1._log_dens(sample)
            ev_log0 = self(param_0)._log_dens(sample)
            eval_log = ev_log1 - ev_log0

            kl = np.mean(eval_log)

            return der_kl, kl

        return fun

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
            :math:`\sum_i(f(proba_1(\phi_i)/proba_0(\phi_i))` with :math:`\phi_i` sampled 
            through proba_0.gen (typically i.i.d.)

        Note:
            For a ProbaMap object obtained as the result of .reparametrize method with
            inherit_method=True, this method might be hidden behind a function attribute which
            is more efficient.
        """
        return self._map(param_1).f_div(self._map(param_0), f=f, n_sample=n_sample)  # type: ignore

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

        proba_0 = self._map(param_0)

        proba_param_shape = self.proba_param_shape
        if proba_param_shape is None:
            raise ValueError("'proba_param_shape' must be specified")

        n_dim = prod(proba_param_shape)

        def gradient_function(
            param_1: ProbaParam, n_sample: int = 1000
        ) -> tuple[ProbaParam, float]:
            log_dens_der = self._log_dens_der(param_1)
            proba_1 = self._map(param_1)

            def to_integrate(samples: Samples) -> np.ndarray:
                out = np.zeros(
                    n_dim + 1
                )  # first n_dim for gradient computation, last dim kl

                delta_log_dens = proba_1._log_dens(samples) - proba_0._log_dens(samples)
                ratio_dens = np.exp(delta_log_dens)  # shape n

                out[:-1] = np.tensordot(  # type: ignore
                    log_dens_der(samples),  # shape (n, proba_param_shape)
                    f_der(ratio_dens),  # shape (n,)
                    (0, 0),
                ).flatten()
                out[-1] = f(ratio_dens) / ratio_dens
                return out

            result = proba_1.integrate(
                to_integrate, n_sample=n_sample, vectorized=True, parallel=False
            )
            return result[:-1].reshape(proba_param_shape), result[-1]

        return gradient_function

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
        return self.grad_f_div(
            param_1, lambda x: x * f(1 / x), lambda x: f(1 / x) - (1 / x) * f_der(1 / x)
        )

    def integrate_der(  # pylint: disable=R0913
        self,
        fun: Callable[[SamplePoint], Union[np.ndarray, float]],
        param: ProbaParam,
        n_sample: int = 1000,
        print_mean: bool = True,
        parallel: bool = False,
    ) -> Tuple[np.ndarray, Union[np.ndarray, float]]:
        r"""
        Compute the derivative of:
        ..math::
            F(alpha) -> \int f(x) \frac{\exp(log p(alpha, x)) d\mu(x)}}{\int \exp(log p(alpha, x)) d\mu(x)}

        Computed through:
        ..math::
            dF = \int f(x) d log p(alpha, x)  dp(alpha, x) - \int d log p(alpha, x)  dp(alpha, x) \int f(x) dp(alpha, x)

        Note: For GD algorithms, integrate_der might not be the best option since constructing large
            samples/ evaluating the function on the sample might be prohibitive. Techniques are
            developped in the PAC-Bayes module to overcome these difficulties.
        """
        proba = self._map(param)
        log_der = self._log_dens_der(param)

        samples = proba(n_sample)

        evals = np.array(
            par_eval(fun, samples, parallel=parallel)
        )  # shape (n, out_shape)
        log_evals = np.array(log_der(samples))  # Shape (n, proba_param_shape)

        mean_eval, sd_eval = evals.mean(0), evals.std(0)

        blab((not print_mean), f"Mean of function: {mean_eval} (deviations: {sd_eval})")

        der = (
            np.tensordot(log_evals, evals - mean_eval, (0, 0)) / n_sample
        )  # shape (proba_param_shape, out_shape)

        return der, mean_eval

    def integrate_vectorized_der(
        self,
        fun: Callable[[Samples], np.ndarray],
        param: ProbaParam,
        n_sample: int = 1000,
        silent: bool = True,
    ):
        r"""
        Compute the derivative of:
        F(alpha) -> \int f(x) exp(log_p(alpha, x)) dmu(x) / \int exp(log_p(alpha, x)) dmu(x)

        Computed through:
        ..math::
            dF = \int f(x) d log_p(alpha, x)  dp(alpha, x) - \int d log_p(alpha, x)  dp(alpha, x) \int f(x) dp(alpha, x)

        Note: For GD algorithms, integrate_der might not be the best option since constructing large
            samples/ evaluating the function on the sample might be prohibitive. Techniques are
            developped in the PAC-Bayes module to overcome these difficulties.

        Contrary to integrate_der, the function fun is assumed to be vectorized and output
        np.ndarray
        """

        proba = self._map(param)
        log_der = self._log_dens_der(param)

        samples = proba(n_sample)

        evals = fun(samples)  # Shape (n, out_shape)
        log_evals = np.array(log_der(samples))  # Shape (n, proba_param_shape)

        mean_eval, sd_eval = evals.mean(0), evals.std(0)

        blab(silent, f"Mean of function: {mean_eval} (deviations: {sd_eval})")

        der = (
            np.tensordot(log_evals, evals - mean_eval, (0, 0)) / n_sample
        )  # shape (proba_param_shape, out_shape)

        return der, mean_eval

    def read_proba(self, path: str, **kwargs) -> Proba:
        """Reads a distribution from a csv file"""
        param = np.genfromtxt(path, **kwargs)
        return self._map(param)

    def reparametrize(  # pylint: disable=R0913
        self,
        transform: Callable[[ProbaParam], ProbaParam],
        der_transform: Callable[[ProbaParam], np.ndarray],
        inv_transform: Optional[Callable[[ProbaParam], ProbaParam]] = None,
        new_ref_param: Optional[ProbaParam] = None,
        proba_param_shape: Optional[Tuple[int, ...]] = None,
        inherit_methods: bool = True,
    ):
        """
        Given a transform B -> A, and assuming that the ProbaMap is parametrized by A,
        constructs a ProbaMap parametrized by B.
        Note: will fail if self.proba_param_shape attribute is None

        Arguments:
            transform, a transform between space B and A. Should be vectorized.
            der_transform, the Jacobian of the transform (der_transform(b) is a np.ndarray of shape
                (b1, ..., bn, a1, ..., am))
            inv_transform (optional), the inverse of transform, from B to A.
            proba_param_shape (optional), the shape of B.
            inherit_methods, whether the kl, grad_kl, grad_right_kl methods of the output uses the
                implementation from the initial ProbaMap object. Default is True.
        Output:
            A distribution map parametrized by B, with map self o transform

        Note on input output shape:
            Noting self.proba_param_shape = (a1, ..., am) and proba_param_shape = (b1, ..., bn),
            transform takes as input arrays of shape (n1, ..., nk, b1, ..., bm) and outputs array
            of shape (n1, ..., nk, a1, ..., an)

            der_transform takes as input arrays of shape (n1, ..., nk, b1, ..., bn) and outputs
            array of shape (n1, ..., nk, b1, ..., bn, a1, ..., am)

            inv_transform takes as input arrays of shape (n1, ..., nk, a1, ..., an) and outpus array
            of shape (n1, ..., nk, b1, ..., bn)

        Notably, if k = 0, then the outputs are of shape (a1, ..., an), (b1,..., bn, a1, ..., am)
            and (b1, ..., bn)  respectively

        Note: inv_transform is only used to assess the default reference parameter
            inherit_methods works by hiding the methods behind reimplemented functions. Not clean,
                but does the trick as far as tested.
        """

        if self.proba_param_shape is None:
            raise ValueError(
                "proba_param_shape attribute must be specified to reparametrize"
            )

        def prob_map(param: ProbaParam) -> Proba:
            return self._map(transform(param))

        indices: list[int] = list(np.arange(-len(self.proba_param_shape), 0))

        def log_dens_der(param: ProbaParam) -> Callable[[Samples], np.ndarray]:
            j_transform = der_transform(param)  # (new_shape,old_shape)
            old_param = transform(param)  # Shape (old_shape)
            old_log_der = self._log_dens_der(
                old_param
            )  # Will output (pre_shape, old_shape)

            def der(samples: Samples) -> np.ndarray:
                return np.tensordot(  # type: ignore
                    old_log_der(samples),  # (pre_shape, old_shape)
                    j_transform,  # (new_shape, old_shape)
                    [indices, indices],
                )  # Shape (pre_shape, new_shape)

            return der

        if (new_ref_param is None) & (inv_transform is not None):
            if self._ref_param is not None:
                try:
                    new_ref_param = inv_transform(self._ref_param)  # type: ignore
                except Exception:  # pylint: disable=W0703
                    warnings.warn("Could not infer the reference parameter")

        if new_ref_param is not None:
            if proba_param_shape is None:
                proba_param_shape = np.array(new_ref_param).shape
            else:
                new_param_shape = np.array(new_ref_param).shape
                check_shape = new_param_shape == proba_param_shape
                if not check_shape:
                    warnings.warn(
                        f""""
                        proba_param_shape indicated is not coherent with ref_param inferred.
                        Using {new_param_shape} as new shape
                        """
                    )
                    proba_param_shape = new_param_shape

        if not inherit_methods:
            return ProbaMap(
                prob_map=prob_map,
                log_dens_der=log_dens_der,
                ref_param=new_ref_param,
                proba_param_shape=proba_param_shape,
                sample_shape=self._sample_shape,
            )

        # Use old kl method
        def kl(
            param_1: ProbaParam,
            param_0: ProbaParam,
            n_sample: int = 1000,
        ):
            return self.kl(
                transform(param_1),
                transform(param_0),
                n_sample=n_sample,
            )

        # Use old grad kl method if possible
        def grad_kl(param_0: ProbaParam):
            par0 = transform(param_0)
            old_der = self.grad_kl(par0)

            def der(param_1: ProbaParam, n_sample: int = 1000):
                par1 = transform(param_1)
                j_transform = der_transform(param_1)  # (new_shape, old_shape)
                grad_kl, kl = old_der(par1, n_sample)  # old_shape

                return (
                    np.tensordot(  # type: ignore
                        j_transform,  # (new_shape, old_shape)
                        grad_kl,  # (old_shape)
                        [indices, indices],  # (new_shape)
                    ),
                    kl,
                )

            return der

        def grad_right_kl(param_1: ProbaParam):
            par1 = transform(param_1)
            old_der = self.grad_right_kl(par1)

            def der(param_0: ProbaParam, n_sample: int = 1000):
                par0 = transform(param_0)
                j_transform = der_transform(param_0)  #  (new_shape, old_shape)
                grad_kl, kl = old_der(par0, n_sample)  # old_shape

                return (
                    np.tensordot(  # type: ignore
                        j_transform,
                        grad_kl,
                        [indices, indices],
                    ),  # new_shape
                    kl,
                )

            return der

        return TransformedProbaMap(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            ref_param=new_ref_param,
            proba_param_shape=proba_param_shape,
            sample_shape=self._sample_shape,
            kl=kl,
            grad_kl=grad_kl,
            grad_right_kl=grad_right_kl,
            f_div=ProbaMap.f_div,
            grad_f_div=ProbaMap.grad_f_div,
            grad_right_f_div=ProbaMap.grad_right_f_div,
        )

    def subset(
        self,
        sub_indexes: list[int],
        default_param: Optional[ProbaParam] = None,
        inherit_methods: bool = True,
    ):
        r"""
        Define a new ProbaMap object from partial ProbaParam object.

        For a distribution map :math:`M:(\theta_1, \dots, \theta_n)-> \mathcal{P}_\theta`,
        output the distribution map
        ..math::
            (\theta_{id_1}, \dots) \mapsto M((\theta_1^*, \dots, \theta_{id_1}, \dots \theta_n^*))
        where :math:`\theta_i^*` are fixed values inferred from default param.

        Exemple: construct a Gaussian map with fixed mean from the standard gaussian map.
        While could this also be achieved through reparametrize function, we can avoid using the
        sparse der_transform matrix

        Arguments:
            sub_indexes: list of indexes settable in the resulting ProbaMap object
            default_param: default parameter values (used for the non settable parameters). Default
                is None, amounts to self.ref_param .
            inherit_methods, whether the kl, grad_kl, grad_right_kl methods of the output uses the
                implementation from the initial ProbaMap object. Default is True.
        Output:
            The distribution map taking a reduced ProbaMap as input.
        """
        if self.proba_param_shape is None:
            raise ValueError("proba_param_shape attribute must be specified to subset")

        if default_param is None:
            default_param = self._ref_param.copy()

        default_param = np.array(default_param).flatten()

        def to_param(par: ProbaParam) -> ProbaParam:
            full_par = default_param.copy()  # type: ignore # No side effect on default_param
            full_par[sub_indexes] = par
            return full_par.reshape(self.proba_param_shape)

        @overload
        def project(pars: ProbaParam) -> ProbaParam:
            ...

        @overload
        def project(pars: ProbaParams) -> ProbaParams:  # type: ignore
            ...

        def project(
            pars: Union[ProbaParam, ProbaParams]
        ) -> Union[ProbaParam, ProbaParams]:
            nd_pars = np.asarray(pars)
            pre_shape = nd_pars.shape[: len(self.proba_param_shape)]

            if pre_shape == ():
                return nd_pars.flatten()[sub_indexes]

            n_proba_param = prod(self.proba_param_shape)
            return nd_pars.reshape(prod(pre_shape), n_proba_param)[
                :, sub_indexes
            ].reshape(pre_shape + (n_proba_param,))

        # Making use of composition here to define new prob_map and log_dens_der
        set_up_param = interpretation(to_param)
        prob_map = set_up_param(self._map)
        super_set_up_mod = post_modif(post_modif(project))
        log_dens_der = super_set_up_mod(set_up_param(self._log_dens_der))

        # Convert reference param
        ref_param = project(default_param)
        if not inherit_methods:
            return ProbaMap(
                prob_map,
                log_dens_der,
                ref_param=ref_param,
                proba_param_shape=ref_param.shape,
                sample_shape=self._sample_shape,
            )

        def kl(
            param_1: ProbaParam,
            param_0: ProbaParam,
            n_sample: int = 1000,
        ):
            return self.kl(
                to_param(param_1),
                to_param(param_0),
                n_sample=n_sample,
            )

            # Use old grad kl method if possible

        def grad_kl(param_0: ProbaParam):
            par0 = to_param(param_0)
            old_der = self.grad_kl(par0)

            def der(param_1: ProbaParam, n_sample: int = 1000):
                par1 = to_param(param_1)
                grad_kl, kl = old_der(par1, n_sample)
                return (project(grad_kl), kl)

            return der

        def grad_right_kl(param_1: ProbaParam):
            par1 = to_param(param_1)
            old_der = self.grad_right_kl(par1)

            def der(param_0: ProbaParam, n_sample: int = 1000):
                par0 = to_param(param_0)
                grad_kl, kl = old_der(par0, n_sample)

                return (project(grad_kl), kl)

            return der

        return TransformedProbaMap(
            prob_map=prob_map,
            log_dens_der=log_dens_der,
            ref_param=ref_param,
            proba_param_shape=ref_param.shape,
            sample_shape=self._sample_shape,
            kl=kl,
            grad_kl=grad_kl,
            grad_right_kl=grad_right_kl,
            f_div=ProbaMap.f_div,
            grad_f_div=ProbaMap.grad_f_div,
            grad_right_f_div=ProbaMap.grad_right_f_div,
        )

    def transform(
        self,
        transform: Callable[[Samples], Samples],
        inv_transform: Callable[[Samples], Samples],
        der_transform: Optional[Callable[[Samples], np.ndarray]] = None,
    ):
        r"""
        Transform the Class of probability :math:`X_\theta \sim \mathbb{P}_{\theta}` to the
        class of probability :math:`transform(X_\theta)`

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
            If proba outputs samples of shape (s1, ..., sk), and transforms maps them to
            (t1, ..., tl),
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

        def new_map(x: ProbaParam) -> Proba:
            return self._map(x).transform(transform, inv_transform, der_transform)

        def new_log_dens_der(
            x: ProbaParam,
        ) -> Callable[[Samples], np.ndarray]:
            log_dens_der_fun = self._log_dens_der(x)

            def new_func(samples: Samples) -> np.ndarray:
                return log_dens_der_fun(inv_transform(samples))

            return new_func

        return TransformedProbaMap(
            prob_map=new_map,
            log_dens_der=new_log_dens_der,
            ref_param=self._ref_param,
            proba_param_shape=self.proba_param_shape,
            sample_shape=self._map(self._ref_param)._sample_shape,
            kl=self.kl,
            grad_kl=self.grad_kl,
            grad_right_kl=self.grad_right_kl,
            f_div=self.f_div,
            grad_f_div=self.grad_f_div,
            grad_right_f_div=self.grad_right_f_div,
        )

    def forget(self):
        """
        Returns a ProbaMap object with standard implementation for methods.
        """
        return ProbaMap(
            prob_map=self._map,
            log_dens_der=self._log_dens_der,
            ref_param=self._ref_param,
            proba_param_shape=self._proba_param_shape,
            sample_shape=self._sample_shape,
        )


class TransformedProbaMap(ProbaMap):
    """Class for transformed ProbaMap. Reimplementations of kl and its derivatives are preserved"""

    def __init__(  # pylint: disable=R0913
        self,
        prob_map: Callable[[ProbaParam], Proba],
        log_dens_der: Callable[[ProbaParam], Callable[[Samples], np.ndarray]],
        ref_param: Optional[ProbaParam] = None,
        proba_param_shape: Optional[tuple[int, ...]] = None,
        sample_shape: Optional[tuple[int, ...]] = None,
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
            ref_param=ref_param,
            proba_param_shape=proba_param_shape,
            sample_shape=sample_shape,
        )

        if kl is None:
            kl = ProbaMap.kl
        self._kl = kl
        if grad_kl is None:
            grad_kl = ProbaMap.grad_kl
        self._grad_kl = grad_kl
        if grad_right_kl is None:
            grad_right_kl = ProbaMap.grad_right_kl
        self._grad_right_kl = grad_right_kl

        if f_div is None:
            f_div = ProbaMap.f_div
        self._f_div = f_div
        if grad_f_div is None:
            grad_f_div = ProbaMap.grad_f_div
        self._grad_f_div = grad_f_div
        if grad_right_f_div is None:
            grad_right_f_div = ProbaMap.grad_right_f_div
        self._grad_right_f_div = grad_right_f_div

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
            :math:`D_f(proba_1, proba_0)` approximated as :math:`\sum_i f(\frac{proba_1(\phi_i)}{proba_0(\phi_i))}`
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


def map_tensorize(*maps: ProbaMap) -> ProbaMap:  # pylint: disable=R0914
    """
    Given a collection of distribution maps

        mu1 -> Proba1, mu2 -> Proba2, ...

    returns the map (mu1, mu2 ... ) -> Proba1 x Proba2 x ...

    For the time being, we consider only distributions on 1D arrays.
    """
    proba_param_shapes = [prob_map.proba_param_shape for prob_map in maps]
    proba_param_len = [
        prod(proba_param_shape) for proba_param_shape in proba_param_shapes
    ]
    proba_param_shape_tot = (sum(proba_param_len),)

    sample_shapes = [prob_map._sample_shape for prob_map in maps]
    if any(sample_shape is None for sample_shape in sample_shapes):
        raise ValueError("All maps to tensorize should have 'sample_shape' information")
    sample_len = [prod(sample_shape) for sample_shape in sample_shapes]  # type: ignore
    sample_shape_tot = (sum(sample_len),)

    pivot_param = np.cumsum([0] + proba_param_len)
    pivot_sample = np.cumsum([0] + sample_len)

    def split_param(param: ProbaParam) -> list[ProbaParam]:
        param = np.array(param)
        return [
            param[p0:p1].reshape(proba_param_shape)
            for p0, p1, proba_param_shape in zip(
                pivot_param[:-1], pivot_param[1:], proba_param_shapes
            )
        ]

    def _decomp_sample(samples: Samples) -> list[Samples]:
        pre_shape = _get_pre_shape(samples, sample_shape_tot)

        samples = samples.reshape((prod(pre_shape), samples.shape[-1]))

        return [
            samples[:, p0:p1].reshape(pre_shape + sample_shape)
            for p0, p1, sample_shape in zip(
                pivot_sample[:-1], pivot_sample[1:], sample_shapes
            )
        ]

    def prob_map(param: ProbaParam) -> Proba:
        params = split_param(param)
        return tensorize(
            *tuple(p_map(par) for par, p_map in zip(params, maps)), flatten=True
        )

    def log_dens_der(param: ProbaParam) -> Callable[[Samples], np.ndarray]:
        params = split_param(param)
        ldds = [p_map._log_dens_der(par) for par, p_map in zip(params, maps)]

        def ldd(samples: Samples) -> np.ndarray:
            samples = np.array(samples)
            pre_shape = samples.shape[:-1]
            n_samples = prod(pre_shape)
            samples = samples.reshape((n_samples, samples.shape[-1]))

            ls_samples = _decomp_sample(
                samples
            )  # list[ Array of shape (n_samples, samp_shape_i)]

            ders = [ldd_i(sample) for sample, ldd_i in zip(ls_samples, ldds)]

            der = np.zeros((n_samples,) + proba_param_shape_tot)
            for p0, p1, der_i in zip(pivot_param[:-1], pivot_param[1:], ders):
                der[:, p0:p1] = der_i.reshape((n_samples, prod(der_i.shape[1:])))

            return der.reshape(pre_shape + proba_param_shape_tot)

        return ldd

    ref_params = [p_map._ref_param for p_map in maps]
    if not np.any([ref_param is None for ref_param in ref_params]):
        ref_param = np.zeros(proba_param_shape_tot)
        for p0, p1, ref in zip(pivot_param[:-1], pivot_param[1:], ref_params):
            ref_param[p0:p1] = ref.flatten()
    else:
        ref_param = None

    proba_map = ProbaMap(
        prob_map=prob_map,
        log_dens_der=log_dens_der,
        ref_param=ref_param,
        proba_param_shape=proba_param_shape_tot,
        sample_shape=sample_shape_tot,
    )

    # Overwrite kl method
    def kl(
        param_1: ProbaParam, param_0: ProbaParam, n_sample: int = 1000
    ):  # pylint: disable=W0613
        param_1_s = split_param(param_1)
        param_0_s = split_param(param_0)

        return np.sum(
            [
                p_map.kl(param_1_i, param_0_i, n_sample)
                for p_map, param_1_i, param_0_i in zip(maps, param_1_s, param_0_s)
            ]
        )

    proba_map.kl = kl  # type: ignore

    def grad_kl(
        param_0: ProbaParam,
    ) -> Callable[[ProbaParam, int], Tuple[ProbaParam, float]]:
        param_0_s = split_param(param_0)
        grad_kls = [
            d_map.grad_kl(param_0_i) for d_map, param_0_i in zip(maps, param_0_s)
        ]

        def der(param_1: ProbaParam, n_sample: int = 1000):
            param_1_s = split_param(param_1)

            info = [
                grad_kl_i(param_1_i, n_sample)
                for grad_kl_i, param_1_i in zip(grad_kls, param_1_s)
            ]

            ders = [x[0].flatten() for x in info]
            kls = [x[1] for x in info]

            der = np.zeros(proba_param_shape_tot)
            for p0, p1, der_i in zip(pivot_param[:-1], pivot_param[1:], ders):
                der[p0:p1] = der_i

            return (der, np.sum(kls))

        return der

    proba_map.grad_kl = grad_kl  # type: ignore

    def grad_right_kl(
        param_1: ProbaParam,
    ) -> Callable[[ProbaParam, int], Tuple[ProbaParam, float]]:
        param_1_s = split_param(param_1)
        grad_kls = [
            d_map.grad_kl(param_1_i) for d_map, param_1_i in zip(maps, param_1_s)
        ]

        def der(param_0: ProbaParam, n_sample: int = 1000):
            param_0_s = split_param(param_0)

            info = [
                grad_kl_i(param_0_i, n_sample)
                for grad_kl_i, param_0_i in zip(grad_kls, param_0_s)
            ]

            ders = [x[0].flatten() for x in info]
            kls = [x[1] for x in info]

            der = np.zeros(proba_param_shape_tot)
            for p0, p1, der_i in zip(pivot_param[:-1], pivot_param[1:], ders):
                der[p0:p1] = der_i

            return (der, np.sum(kls))

        return der

    proba_map.grad_right_kl = grad_right_kl  # type: ignore

    return proba_map
