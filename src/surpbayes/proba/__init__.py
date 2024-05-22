r"""
Probability sub module

Define classes for probability distribution (Proba) and parametric families of probability
distributions (ProbaMap). Subclasses for specific distributions as well as basic operations for
these classes are defined.

This sub module does not rely on other aduq sub modules. It is used extensively in sub modules
'bayes', 'meta_bayes', and as such in sub modules 'pyadm1' and 'pyam2'.

I. Proba class and inherited classes
I.A. Proba class
I.A.a. Initialization
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

I.A.b Attributes
- gen, see initialization
- log_dens, see initialization
- sample_shape, see initialization
- _sample_size, the size of a single sample (integer)

I.A.c. Methods
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

I.A.d Functions manipulating proba objects
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

I.B. Classes inheriting from Proba
I.B.a. Gaussian
Class for standard multivariate gaussians. Initialisation is modified to be defined from a mean and
covariance attribute. Additional information can be passed to speed up computation.
Additional attributes are:
- "means", the means of the distribution
- "cov", the covariance of the distribution,
- "inv_cov", the inverse of the covariance,
- "vals", the eigenvalues of cov
- "vects", the eigenvectors of cov
Methods 'reshape', 'flatten', 'shift', 'contract' and 'lin_transform' are reimplemented so that they
output Gaussian objects.

I.B.b. BlockDiagGauss
Class for multivariate Gaussians with a block diagonal covariance matrix.

I.B.c. TensorizedGaussian
Class for multivariate Gaussian with diagonal covariance matrix.

II. ProbaMap class and inherited clas
II. A. ProbaMap
II. A. a. Initialization
A ProbaMap object is initialized from 5 arguments (3 optionals):
- prob_map: function which takes as input a ProbaParam (an array like) and outputs a Proba object.
- log_dens_der: function which takes as input a ProbaParam and outputs a closure. This closure
takes as input Samples and returns the derivative of log densities of the distribution mapped by
the ProbaParam, the derivative being with respect to the ProbaParam. Mathematically, it is the
derivative of the function
        $theta, x \rightarrow prob_map(theta).log_dens(x)$
with respect to $\theta$.
- ref_param: reference ProbaParam. Optional.
- proba_param_shape: shape of ProbaParam objects accepted by prob_map. Optional.
- sample_shape: shared sample shape of the probability distributions outputed by
    prob_map. Optional

II. A. b. Attributes
- map: see prob_map in initialization
- log_dens_der: see initialization
- ref_param: see initialization. If not given, is inferred from proba_param_shape (array full of 0
is tried).
- proba_param_shape: see initialization. If not given, tries inferring from ref_param.
- sample_shape: see initialization. If not given, tries inferring from ref_param and prob_map
- _sample_size: size of a single sample outputed by any distribution

II. A. c. Methods
- __call__: uses map attribute
- kl: computes the Kullback-Leibler divergence between two probability distributions encoded as
ProbaParam. A number of samples to use is also taken as input.
- grad_kl: computes the gradient of Kullback-Leibler divergence with respect to the left
probability distribution. Takes the right probablity distribution as input and returns a closure.
- grad_right_kl: computes the gradient of Kullback-Leibler divergence with respect to the right
probability distribution. Takes the left probablity distribution as input and returns a closure.
- f_div: computes the f-divergence between two probability distributions encoded as
ProbaParam. A number of samples to use is also taken as input.
- grad_f_div: computes the gradient of the f-divergence with respect to the left probability
distribution. Takes the right probablity distribution as input and returns a closure.
- grad_right_f_div: computes the gradient of the f-divergence with respect to the right probability
distribution. Takes the left probablity distribution as input and returns a closure.
- integrate_der: computes the derivative of the expected value of a function with respect to a
ProbaParam.
- read_proba: reads a csv file containing a ProbaParam and outputs a Proba object.
- reparametrize: returns a new ProbaMap describing the same Proba objects but using a different
parametrisation.
- subset: construct a new ProbaMap from the ProbaMap by fixing some of the ProbaParam values.
- transform: construct a ProbaMap describing g(X) where X is a Proba from the original ProbaMap
- forget: construct a new ProbaMap with default implementations for all methods.

II. A. d. Method inheritance
A key feature of the ProbaMap class is method inheritance. For probability distributions X, Y,
quantities such as f divergence (and in particular Kullback-Leibler divergence) are left invariant
by the transform $X,Y \rightarrow g(X), g(Y)$ if $g$ is inversible. As such, reimplemented methods
such as kl, grad_kl, grad_right_kl are inhereted from the initial implementation after transform,
that is to say. To do that, the methods are hidden behind attributes (ugly but effective). This is
automatically done for transform. Methods can also be inherited when using subset and reparametrize.

II. B. Classes inherited from ProbaMap
II. B. a. ExponentialFamily
Class for Exponential Family using the natural parametrisation. Initialization is modified to take
as input a sampling function, the feature map, normalisation function, reference log density and
derivatives of these functions. Kullback Leibler related methods are reimplemented. transform is
also reimplemented and output an ExponentialFamily object.

Future feature: subset method should be reimplemented to output an ExponentialFamily

II. B. b. PreExpFamily
Class for Exponential Families not using the natural parametrisation. Initialization is modified to
take as input map to Proba objects, the feature map, the map to the natural parametrisation,
its inverse and derivative of the inverse. The transform method is reimplemented to output a
PreExpFamily object.

II. B. c. GaussianMap
Class for Gaussian distributions, inherited from PreExpFamily. Initialisation is modified to take as
input only sample_size and sample_shape (only one of those needs to be specified).
KL method and related are reimplemented.

II. B. d. BlockGaussianMap
Class for Gaussian distributions with Block Diagonal covariance (fixed blocks), inherited from
PreExpFamily. Initialisation is modified to take as input only a list of block (as a list of list
of int). KL method and related are reimplemented.

II. B. e. TensorizedGaussianMap
Class for Gaussian distributions with Diagonal covariance, inherited from PreExpFamily.
Initialisation is modified to take as input only sample_size and sample_shape (only one of those
needs to be specified). KL method and related are reimplemented.

About vectorization
1. Which input functions should be vectorized?
Vectorization is assumed for transforms and internal functional attributes log_dens, log_dens_der
but it is not assumed for outside functions. That is to say, integrate and integrate_der expect
not vectorized function, all remaining functions are expected to be vectorized.

2. What do you mean by vectorization?
Vectorized is shape smart and is done on the left. i.e, if the base function expects inputs of shape
(s1, ..., sn) and outputs (o1, ...., om), then the vectorized function is meant to take inputs of
shape (shape_before, s1, ..., sn) to which it outputs (shape_before, o1, ..., om).

3. How can I vectorize my function?
Quite a lot of functions are "natively" vectorized, though perhaps not using the same convention
(i.e. witness lambda x : a @ x). The first thing should be to change the implementation if possible
(e.g. lambda x: np.tensordot(x, a, (-1, -1))). If that is not possible, then one can use the
vectorize function from aduq.misc

For future considerations:
- Minimize function recursion added time when transforming.
- Potential speed gain when the transform functions are vectorized (i.e. for linear transforms,
    component wise transforms, etc...). Samples should be passed as sample_shape + (sample_size,)
    for these functions, contrary to current (sample_size,) + sample_shape. Use case: Uniform
    priors through Gaussians (spicy functions being vectorized).
- Notably, this question of vectorization is most important for log_dens evaluations, since many
    routines rely on list comprehension for evaluations of log densities at potentially many
    sample points. Moreover, most users should not consider implementing log_dens themselves but
    rather rely on already implemented distributions/distributions map and then use transforms.
"""

from ..types import ProbaParam, SamplePoint
from ._errors import RenormError
from .exponential_family import ExponentialFamily, PreExpFamily
from .gauss import (
    BlockDiagGauss,
    BlockDiagGaussMap,
    FactCovGaussianMap,
    FixedCovGaussianMap,
    GaussHypercubeMap,
    Gaussian,
    GaussianMap,
    TensorizedGaussian,
    TensorizedGaussianMap,
)
from .proba import Proba, add, from_sample, mixture, tensorize
from .proba_map import ProbaMap, map_tensorize
