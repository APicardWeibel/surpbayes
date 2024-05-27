r"""
Modular way to perform Catoni's optimisation task using SurPAC algorithm for exponential
families, in particular gaussian distributions and related.

The rationale behind score optimisation for exponential families is the closed form prior to
posterior transform when the score belongs to a class of functions naturally defined from the
exponential family. Moreover, it can be shown that the output of Catoni's minimisation is not
changed by using the best L2 approximation of the score at the posterior distribution. Hence
the algorithm alternates score approximation from posterior approximation and posterior
approximation from score approximation until convergence.

The resulting algorithm is much more stable than the gradient descent based one, especially when
the score belongs to the function class (i.e., for gaussians, is a quadratic form). To further
increase stability, the integration approximation scheme is changed to take into account most of
the previous draws, by giving them weights based on a partition of the space where each set is
'centered' around a specific draw. The gaussian case is treated separately, as each distribution
induces a natural metric, used when defining the partition of the space.

To avoid relying too much on a partial description of the score, two concurrent mechanisms are
implemented:
- Rather than defining the posterior parameter as :math:`prior\_param - \lambda^{-1} score\_approx`, the
update rule is dampened to
..math::
    post\_param + (1-dampen) (prior\_param - \lambda^{-1} score\_approx - post\_param)
- dampen is chosen so that: :math:`(1- dampen) < (1- dampen\_min)`
and :math:`KL(new\_post\_param, post\_param)< kl\_max`

Exponential families of particular interest here are 'modal ones' (i.e. gaussians, beta).

The routine is written so that it can be used for the following distribution classes (from proba
module):
- 'GaussianMap'
- 'BlockDiagGaussMap'
- 'TensorizedGaussianMap'
- 'PreExpFamily'
- 'ExponentialFamily'

As mentionned above, Gaussian distributions are treated somewhat differently, since they induce a
natural metric, have easy to compute high confidence region, and have easy to compute log density
derivative with respect to the sample point x. All these properties can be used to improve the
standard methods used to generate new samples (i.e. using SVGD like algorithms) and obtain weights
(see weighing submodule).

The 'PreExpFamily' and 'ExponentialFamily' rely on near duplicate implementations (changes due to
different attribute names). The only conceptual difference lies when enforcing that
:math:`KL(new\_post\_param, post\_param)< kl\_max`. In the case of exponential families, it can be
supposed that the function :math:`alpha \rightarrow KL(post\_param + alpha dir, post\_param)` is non
decreasing, which is no longer the case for 'PreExpFamily' as the natural parametrisation is not
used. This could lead to suboptimal choices of :math:`\alpha` and potentially to instabilities.

Remark:
    The key observation here is the conjugation between certain type of functions and priors.
It is possible that some of the results here could be generalised to other types of
prior/posterior conjugation known.
    The exponential family framework used here can be used for transform of exponential families.

TODO:
- Propose new KL solver using derivative and second order derivative if available
"""

from .accu_sample_exp import AccuSampleValExp
from .gauss_solver import GaussianSPACS
from .pre_exp_solver import PreExpSPACS
from .surpac_solver import SurPACSolver
