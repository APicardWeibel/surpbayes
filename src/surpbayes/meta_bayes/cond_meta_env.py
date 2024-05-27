"""Variant of MetaLearningEnv for conditional setting

So far, computations seem to exhibit much more instability than unconditional metalearning settings.
"""
from typing import Callable

import numpy as np
from surpbayes.bayes import pacbayes_minimize
from surpbayes.meta_bayes.hist_meta import HistMeta
from surpbayes.meta_bayes.meta_env import MetaLearningEnv
from surpbayes.meta_bayes.task import Task
from surpbayes.misc import blab
from surpbayes.proba import ProbaMap
from surpbayes.types import MetaData, MetaParam, ProbaParam


class CondMetaEnv(MetaLearningEnv):
    r"""Conditional Meta Learning environnement for minimisation of Catoni Pac Bayes bound

    For a collection of task, meta learns a suitable map from metadata to prior.

    Class attributes:
    - proba_map: A ProbaMap instance, defining the shared family of probabilities in which the meta
    prior is learnt, and in which the tasks' posterior live.
    - list_tasks: A list of Task objects. All tasks should share the same parameter space, coherent
    with the proba_map attribute (this is not checked, training will fail).
    - meta_param: MetaParam, containing the current meta parameter.
    - cond_map, the map from task meta data and meta parameter to a ProbaParam
    - der_cond_map, the derivative of cond_map with respect to the meta parameter
    - hyperparams: dictionary, containing hyperparameters for training each task.
    - hist_meta: HistMeta, track the evolution of score and prior_param during training
    - n_task: int, the number of tasks
    - task_score: the list of end penalised score for each task
    - converged: boolean, specifying if convergence has been reached
    - meta_score: float, the current meta score for the prior


    Routine motivation:
    In the context of penalised risk minimisation inner algorithm, the meta gradient is easy to
    compute (see below). As such, the meta training algorithm is a Gradient descent procedure. To
    improve stability, the prior distribution is forced to evolve slowly (in term of KL divergence)

    Gradient of the meta score for Catoni Pac-Bayes.
    For a proba map :math:`\pi`, noting :math:`\theta_0` the prior parameter, :math:`R_i`,
    :math:`\lambda_i` the empirical risk function and temperature for task i,
    :math:`\hat{\theta}_i = \hat{\theta}_i(\theta_0)` the posterior parameter using prior
    :math:`\theta_0`, the meta score of prior parameter :math:`\theta_0` is
    defined as :math:`S(\theta_0) = \sum_i S_i(\theta_0)` where
    ..math::
        S_i(\theta_0)
        = \pi(\hat{\theta}_i)[R_i] + \lambda_i KL(\pi(\hat{\theta}_i), \pi(\theta_0))

    In the context of conditional meta learning, the prior is constructed from a meta parameter,
    which is learnt, and taks meta data :math:`m_i`. As such, the meta score of meta param
    :math:`\alpha` is defined as:
    ..math::
        S(\alpha) = \sum_i S_i(T(\alpha, m_i))

    where T is the map from meta data to prior parameter. Noting :math:`\theta_i = T(\alpha, m_i)`,
    the derivative of :math:`S_i(\theta_i)` with respect to :math:`\theta_i` has simple expression
    :math:`\lambda_i K_i` where :math:`K_i` is the gradient of the Kullback--Leibler term
    :math:`KL(\pi(\hat{\theta}_i), \pi(\theta_i))` with respect to :math:`\theta_i` at fixed :math:`\hat{\theta}_i`
    value. Therefore, the gradient of :math:`S_i(T(\alpha, m_i))` with respect to :math:`\alpha` can be
    computed using the chain rule.
    """

    def __init__(
        self,
        proba_map: ProbaMap,
        list_task: list[Task],
        cond_map: Callable[[MetaParam, MetaData], ProbaParam],
        der_cond_map: Callable[[MetaParam, MetaData], np.ndarray],
        meta_param: MetaParam,
        **hyperparams,
    ):
        """Initialize conditional meta learning environnement.

        Args:
            proba_map (ProbaMap): class of distributions on which priors/posterior are optimized
            list_task (list[Task]): list of learning task constituing the meta learning
                environnement
            cond_map (Callable): map task meta data and meta parameter to a ProbaParam
            der_cond_map (Callable): derivative of cond_map with respect to the meta parameter
            meta_param (MetaParam): initial Meta parameter value.
            **hyperparams (dict): further arguments passed to pacbayes_minimize (inner
                learning algorithm).
        """
        super().__init__(proba_map=proba_map, list_task=list_task, **hyperparams)

        self.meta_param = np.array(meta_param)

        self.cond_map = cond_map
        self.der_cond_map = der_cond_map

        self._meta_shape = self.meta_param.shape
        self.hist_meta = HistMeta(
            meta_param_shape=self._meta_shape, n=1, n_task=self.n_task
        )
        self.hist_meta.add1(self.meta_param, np.nan, self.task_score)  # type: ignore

        self._prob_ndim = len(proba_map.proba_param_shape)

    def train(self, task: Task, **hyperparams) -> None:
        """Perform inner learning for a task using learning environnement prior.

        Posterior and the accu sample val are update in place in the task.

        The inner algorithm called is 'aduq.bayes.pacbayes_minimize.' The routine used depends
        on the proba_map and hyperparams attributes of the learning environnement (pre inferred
        at construction time).

        The 'accu_sample_val' field of the task is indirectly augmented by pacbayes_minimize.
        """
        if hyperparams:
            loc_hyperparams = self.hyperparams.copy()
            loc_hyperparams.update(hyperparams)
        else:
            loc_hyperparams = self.hyperparams

        # Infer the prior_param from the conditional mapping with meta_param
        prior_param = self.cond_map(self.meta_param, task.meta_data)

        # Perform the inner algorithm
        opt_res = pacbayes_minimize(
            fun=task.score,
            proba_map=self.proba_map,
            prior_param=prior_param,
            post_param=task.post_param,
            temperature=task.temp,
            prev_eval=task.accu_sample_val,
            vectorized=task.vectorized,
            parallel=task.parallel,
            **loc_hyperparams,
        )

        # Store output in task
        task.post_param = opt_res.opti_param
        task.end_score = opt_res.opti_score  # type: ignore

    def grad_meta(self, task: Task, n_grad_KL: int = 10**4) -> ProbaParam:
        """Compute the meta gradient for a provided task.

        Arg:
            task: a Task object.

        Output:
            The gradient of the penalised meta score with respect to the meta parameter.
        """
        # Perform the inner algorithm
        self.train(task)

        # Recompute the prior parameter
        prior_param = self.cond_map(self.meta_param, task.meta_data)  # type: ignore

        # Compute the gradient of the meta parameter as temp * J meta_to_prior @ nabla_2 KL

        return task.temp * np.tensordot(
            self.der_cond_map(self.meta_param, task.meta_data),
            self.proba_map.grad_right_kl(task.post_param)(prior_param, n_grad_KL)[0],  # type: ignore
            (
                tuple(range(-self._prob_ndim, 0)),
                tuple(range(self._prob_ndim)),
            ),
        )

    def _get_eta_use(self, grad: np.ndarray, kl_max: float, eta_loc: float):
        """De-activated for now - returns eta_loc"""
        return eta_loc

    def meta_learn(
        self,
        epochs: int = 1,
        eta: float = 0.01,
        kl_max: float = 1.0,
        mini_batch_size: int = 10,
        silent: bool = False,
    ) -> None:
        """Meta Learning algorithm

        Args:
            epochs (int): number of learning epochs (default 1)
            eta (float): step size for gradient descent (default 0.01)
            kl_max (float): Currently disregarded (kept for compatibility with meta_env)
            mini_batch_size (int): size of mini batches.
            silent (bool): should there be any print

        Outputs:
            None (modifications inplace)

        The tasks are read one after another and the meta_param is updated after each task is read.
        Difference with MetaLearnEnv: the meta_param rather than prior_param is updated
        """
        # Extend history for meta learning log
        self.hist_meta.extend_memory(epochs)

        # Extend memory for tasks (once and for all rather than iteratively)
        [
            self._extend_memo(task, epochs) for task in self.list_task
        ]  # pylint: disable=W0106

        # Define step size
        eta_loc = eta / self.n_task
        batch_count = (self.n_task // mini_batch_size) + (
            (self.n_task % mini_batch_size) > 0
        )
        # Main learning loop
        for i in range(epochs):
            blab(silent, f"Iteration {i+1}/{epochs}")

            permut = np.random.permutation(self.n_task)

            for n_batch in range(batch_count):
                blab(silent, f"Starting minibatch {n_batch+1}")
                grad = self._init_grad()
                start = mini_batch_size * n_batch
                iloc_task_s = permut[start : (start + mini_batch_size)]
                for j, iloc_task in enumerate(iloc_task_s):
                    task = self.list_task[iloc_task]
                    blab(
                        silent, f"Starting task {iloc_task} ({start+j+1}/{self.n_task})"
                    )
                    grad = grad - self.grad_meta(task)
                    self.task_score[iloc_task] = task.end_score

                blab(
                    silent,
                    f"Minibatch {n_batch + 1} avg score: {self.task_score[iloc_task_s].mean()}",
                )

                eta_use = self._get_eta_use(grad, kl_max, eta_loc)
                self.meta_param = self.meta_param - eta_use * grad

                blab(silent)

            # Log meta learning result
            self.meta_score = np.mean(self.task_score)
            self.hist_meta.add1(self.meta_param, self.meta_score, self.task_score)

            blab(silent, f"Meta score: {self.meta_score}\n")

    def meta_learn_batch(
        self, epochs=1, eta=0.01, kl_tol=10**-3, kl_max=1.0, silent=False
    ):  # pylint: disable=W0221
        """
        Meta Learning algorithm (batch variant)

        Args:
            epochs (int): number of learning epochs (default 1)
            eta (float): step size for gradient descent (default 0.01)
            kl_tol, kl_max (float): Disregarded
            silent (bool): should there be any print

        Outputs:
            None (modifications inplace)

        The prior is updated after all tasks have been read. Improves stability at the cost of
        duration (for the early stages) compared to non batch version.
        """
        # Extend history for meta learning log
        self.hist_meta.extend_memory(epochs)

        self.extend_tasks_memo(epochs)

        # Define step size
        eta_loc = eta / self.n_task

        # Main learning loop
        for i in range(epochs):
            blab(silent, f"Iteration {i}")
            # Prepare accu for gradient
            grad = np.zeros(self._meta_shape)

            # Iterate over tasks
            for j, task in enumerate(self.list_task):
                blab(silent, f"Starting task {j}")
                # Compute gradient (this updates task posterior automatically)
                grad = grad - self.grad_meta(task)

                # Store end score for task
                self.task_score[j] = task.end_score

            # Compute new meta param
            new_meta_param = self.meta_param + eta_loc * grad

            # Log/update meta learning result
            self.meta_param = new_meta_param
            self.meta_score = np.mean(self.task_score)
            self.hist_meta.add1(self.meta_param, self.meta_score, self.task_score)

            blab(silent, f"Meta score: {self.meta_score}\n")
