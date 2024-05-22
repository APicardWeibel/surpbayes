r"""
Bayesian flavored meta learning sub module.

Class and functions designed to Meta Learn a prior distribution in the context of Pseudo Gibbs
inner algorithm.

I. Meta-Learning for Penalized algorithm
Meta Learning is a way to learn inductive bias, in order to speed up learning or obtain better
performances when data is scarce.

A simple way to add inductive bias to the classic empirical risk minimisation learning algorithm is
to add a penalisation term. As such, for a risk $\hat{R}$, the calibrated parameter is defined as

    $$\hat\theta \in \arg\min \hat{R}(\theta) + \lambda P(\theta, \theta_0) $$

where $P$ is a penalisation function and $\lambda$ controlling the learning rate.

In a meta learning context, one has a variety of tasks, which we can sum up in a collection of
empirical risks $\hat{R}_i$. We also consider different learning rates $\lambda_i$ for each task.

For a given inductive bias $\theta_0$, which we will name meta parameter from now on, one can
consider the penalized meta score

    $$ S(\theta_0) = \sum_i \hat{R}_i(\hat{\theta}_i) + \lambda_i P(\hat{\theta}_i, \theta_0) $$
where $\hat{\theta}_i$ is the minimizer of the penalized risk, and as such is a function of
$\theta_0$.

Assuming that the score and penalisation are differentiable, then the meta score derivative can be
computed as:

    $$ \nabla S(\theta_0) = \sum_i \lambda_i \nabla_2 P(\hat{\theta}_i, \theta_0). $$

This formula relies on the fact that the penalised score is minimized at $\hat{\theta}_i$, and
therefore the gradient of the penalised score with respect to $\hat{\theta}_i$ is $0$.

Optimisation of this criteria is straightforward, as it can alternate between calibration for each
task (learn $\hat{\theta}_i$ from current $\theta_0$) and correcting the value of $\theta_0$ using
gradient descent after calibration
($\theta_0 = \theta_0 -\eta\lambda_i \nabla_2 P(\hat{\theta}_i, \theta_0)$).

II. Transcription to PAC-Bayesian algorithms
PAC-Bayesian algorithms fit into the framework described above, the posterior being defined as the
minimizer of a penalized bound. In this setting, the gradient descent algorithm involves the
computation of the derivative of Kullback-Leibler divergence with respect to the second
probability, for which there exists closed form expressions for exponential families.

The PAC-Bayesian implementation strives to make the most use of each evaluation of the score for
each task - being the bottleneck for tasks dealing with the calibration of complex models. This is
performed at the inner algorithm level, by keeping track of all previous evaluations and using them
to efficiently compute integrals. We remark that the efficiency of the algorithm is greatly
influenced by the efficiency of the inner learning algorithm.

III. Future
- Deal with cases where memory becomes an issue (load/save/forget task data iteratively)
- Conditional Meta Learning for tasks with distinct ProbaMap.
"""

from surpbayes.meta_bayes.hist_meta import HistMeta
from surpbayes.meta_bayes.meta_env import MetaLearningEnv
from surpbayes.meta_bayes.task import Task
