import numpy as np

from surpbayes.meta_bayes.task import Task
from surpbayes.proba import ProbaMap
from surpbayes.bayes import pacbayes_minimize
from surpbayes.types import ProbaParam, ProbaParams

from surpbayes.misc import par_eval, blab


def test_eval_task(
    meta_param: ProbaParam,
    test_task: Task,
    proba_map: ProbaMap,
    n_test: int,
    hyperparams: dict,
):
    """Evaluate a test task at meta_param

    The task training is performed from scratch (empty accu),
    starting from meta_param prior. This prevents potential bias due
    to more accurate solutions being found after some iterations.
    """
    opt_res = pacbayes_minimize(
        fun=test_task.score,
        proba_map=proba_map,
        prior_param=meta_param,
        post_param=meta_param,
        temperature=test_task.temp,
        prev_eval=None,
        vectorized=test_task.vectorized,
        parallel=test_task.parallel,
        **hyperparams,
    )
    post_param = opt_res.opti_param

    post = proba_map(post_param)
    if test_task.vectorized:
        mean_score = np.mean(test_task.score(post(n_test)))
    else:
        mean_score = np.mean(
            par_eval(test_task.score, post(n_test), parallel=test_task.parallel)
        )

    return mean_score + test_task.temp * proba_map.kl(post_param, meta_param)


def eval_meta_param(
    meta_param: ProbaParam,
    test_tasks: list[Task],
    proba_map: ProbaMap,
    n_test: int,
    hyperparams,
    silent: bool = False,
):
    """Evaluate a meta_param on a list of test_tasks"""
    accu = np.zeros(len(test_tasks))
    for i, task in enumerate(test_tasks):
        perf = test_eval_task(
            meta_param=meta_param,
            test_task=task,
            proba_map=proba_map,
            n_test=n_test,
            hyperparams=hyperparams,
        )
        blab(silent, f"Task {i}: {perf}")
        accu[i] = perf
    return accu


def eval_meta_hist(
    meta_params: ProbaParams,
    test_tasks: list[Task],
    proba_map: ProbaMap,
    n_test: int = 100,
    hyperparams: dict = {},
    silent: bool = False,
):
    """Evaluate a succession of meta_params"""

    accu = np.zeros((len(meta_params), len(test_tasks)))
    for j, meta_param in enumerate(meta_params):
        blab(silent, f"Starting meta_param {j}")
        accu[j] = eval_meta_param(
            meta_param=meta_param,
            test_tasks=test_tasks,
            proba_map=proba_map,
            n_test=n_test,
            hyperparams=hyperparams,
            silent=silent,
        )
        blab(silent, f"meta_param {j} perf: {np.mean(accu[j])}")
    return accu
