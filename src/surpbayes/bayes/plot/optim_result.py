from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

from surpbayes.bayes.optim_result_bayes import OptimResultBayes
from surpbayes.bayes.surpac.weighing import get_weights_mc


def plot_score_evol(
    opt_res: OptimResultBayes,
    n_sample_weight_estim=10**5,
    n_y=400,
    cmap="RdBu",
    bw_method=None,
    plot=plt,
):
    """
    Plot the evolution of the score pushforward of the posterior as the posterior is trained

    Note that this plot takes quite a while to be generated.
    """
    sample_val = opt_res.sample_val
    proba_map = opt_res.log_bayes.proba_map
    proba_pars = opt_res.log_bayes.proba_pars()

    y_min, y_max = (
        np.min(sample_val.vals()),
        np.max(sample_val.vals()),
    )

    d_y = y_max - y_min
    delta_y = d_y / sample_val.n_filled
    y_min, y_max = y_min - delta_y, y_max + delta_y

    if proba_map.map_type == "Gaussian":
        proba_end = proba_map(proba_pars[-1])
        vals, vects = proba_end.vals, proba_end.vects  # type: ignore
        half_metric_matrix = ((1 / np.sqrt(vals)) * vects) @ vects.T
    else:
        half_metric_matrix = None

    weights = [
        get_weights_mc(
            proba_map(proba_par),
            sample_val.sample(),
            half_metric_matrix=half_metric_matrix,
            n_sample_estim=n_sample_weight_estim,
        )
        for proba_par in proba_pars
    ]

    ys = np.linspace(y_min, y_max, n_y)
    values = sample_val.vals()

    def get_densities(w):
        gkde = gaussian_kde(values, bw_method=bw_method, weights=w)
        return gkde.pdf(ys)

    densities = np.array([get_densities(w) for w in weights])

    n_x = densities.shape[0]
    xs = np.full((n_y + 1, n_x + 1), np.arange(n_x + 1)).T
    ys = np.full((n_x + 1, n_y + 1), np.linspace(y_min, y_max, n_y + 1))

    fig, ax = plot.subplots()

    c = ax.pcolormesh(
        xs, ys, densities, cmap=cmap, vmin=np.min(densities), vmax=np.max(densities)
    )
    ax.set_title("Evolution of score densities")
    # set the limits of the plot to the limits of the data
    ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])
    fig.colorbar(c, ax=ax)

    plot.xlabel("Step")
    plot.ylabel("Score")
    return plot


def plot_score_push_begin_end(
    opt_res: OptimResultBayes,
    n_sample_weight_estim: int = 10**5,
    bw_method=None,
    plot=plt,
    n_y: int = 400,
):
    """Plot pushforward of score for the first and last distribution of an OptimResultBayes"""
    sample_val = opt_res.sample_val
    proba_map = opt_res.log_bayes.proba_map
    proba_pars = opt_res.log_bayes.proba_pars()

    y_min, y_max = (
        np.min(sample_val.vals()),
        np.max(sample_val.vals()),
    )

    d_y = y_max - y_min
    delta_y = d_y / sample_val.n_filled
    y_min, y_max = y_min - delta_y, y_max + delta_y

    ys = np.linspace(y_min, y_max, n_y)

    if proba_map.map_type == "Gaussian":
        proba_end = proba_map(proba_pars[-1])
        vals, vects = proba_end.vals, proba_end.vects
        half_metric_matrix = ((1 / np.sqrt(vals)) * vects) @ vects.T
        hmm = True
    else:
        hmm = False

    if hmm:
        tree = cKDTree(sample_val.sample() @ half_metric_matrix)
    else:
        tree = cKDTree(sample_val.sample())

    if hmm:

        def get_weights(proba):
            large_sample = proba(n_sample_weight_estim)
            min_ids = tree.query(large_sample @ half_metric_matrix)[1]

            counter = Counter(min_ids)

            for i in range(sample_val.n_filled):
                if i not in counter.keys():
                    counter[i] = 0

            return np.array(sorted(counter.items()))[:, 1] / n_sample_weight_estim

    else:

        def get_weights(proba):
            large_sample = proba(n_sample_weight_estim)
            min_ids = tree.query(large_sample)[1]

            counter = Counter(min_ids)

            for i in range(sample_val.n_filled):
                if i not in counter.keys():
                    counter[i] = 0

            return np.array(sorted(counter.items()))[:, 1] / n_sample_weight_estim

    def get_densities(w):
        gkde = gaussian_kde(sample_val.vals(), bw_method=bw_method, weights=w)
        return gkde.pdf(ys)

    plot.plot(
        ys, get_densities(get_weights(proba_map(opt_res.hist_param[0]))), label="Ini."
    )
    plot.plot(
        ys, get_densities(get_weights(proba_map(opt_res.hist_param[-1]))), label="End."
    )
    plot.title("Densities of initial and final distribution")
    plot.xlabel("Score")
    plot.ylabel("Density")
    plot.legend()
    return plot
