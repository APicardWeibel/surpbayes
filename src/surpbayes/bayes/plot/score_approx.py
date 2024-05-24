import matplotlib.pyplot as plt
import numpy as np

from surpbayes.bayes.optim_result_bayes import OptimResultBayes
from surpbayes.bayes.score_approx.weighing import (
    get_weights_mc,
    get_weights_mc_gauss,
)


def plot_weight_per_gen(
    opt_res: OptimResultBayes,
    *args,
    n_sample_estim_weight: int = 10**5,
    plot=plt,
    **kwargs
):
    """Plot evolution of weight per generation"""
    proba_map = opt_res.log_bayes.proba_map
    is_gauss = proba_map.map_type == "Gaussian"
    proba = proba_map(opt_res.opti_param)

    if is_gauss:
        w = get_weights_mc_gauss(
            proba,  # type: ignore
            samples=opt_res.sample_val.sample(),
            n_sample_estim=n_sample_estim_weight,
        )
    else:
        w = get_weights_mc(
            proba,
            samples=opt_res.sample_val.sample(),
            n_sample_estim=n_sample_estim_weight,
        )

    generations = opt_res.sample_val.gen_tracker()
    gen_max = generations[0]
    generations = gen_max - generations
    weight_per_gen = [np.sum(w[generations == k]) for k in range(gen_max + 1)]
    plot.plot(np.arange(gen_max + 1), weight_per_gen, *args, **kwargs)
    plot.xlabel("Step")
    plot.ylabel("Mass")
    plot.title("Repartition of mass between generations for posterior")
    return plot


def plot_weights_per_gen(opt_res: OptimResultBayes, n_sample_estim_weight: int = 10**5):
    proba_map = opt_res.log_bayes.proba_map
    is_gauss = proba_map.map_type == "Gaussian"
    proba = proba_map(opt_res.opti_param)

    if is_gauss:
        w = get_weights_mc_gauss(
            proba,  # type: ignore
            samples=opt_res.sample_val.sample(),
            n_sample_estim=n_sample_estim_weight,
        )
    else:
        w = get_weights_mc(
            proba,
            samples=opt_res.sample_val.sample(),
            n_sample_estim=n_sample_estim_weight,
        )

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_xlabel("Step")
    ax.set_ylabel("Mass")

    generations = opt_res.sample_val.gen_tracker()
    gen_max = generations[0]
    generations = gen_max - generations
    n_per_gen = [np.sum(generations == k) for k in range(gen_max + 1)]
    delta_xs = [1.0 / n_per_gen[k] for k in generations]
    xs = np.cumsum(delta_xs)
    ax2.scatter(xs, w, marker="x", s=0.1)
    weight_per_gen = [np.sum(w[generations == k]) for k in range(gen_max + 1)]

    def mumble(ls):
        acc = []
        for x in ls:
            acc.append(x)
            acc.append(x)
        return acc

    main_x = [0] + mumble(range(1, gen_max + 1)) + [gen_max + 1]
    ax.plot(main_x, mumble(weight_per_gen), color="black", linewidth=1.0)
    return fig, ax
