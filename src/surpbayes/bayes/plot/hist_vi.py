import matplotlib.pyplot as plt

from surpbayes.bayes.hist_vi import HistVILog


def plot_hist_vi(hist_vi: HistVILog, plot=plt):
    """
    Plot evolution of KPIs of a variational inference optimisation task.
    KPIs plotted are the VI score, the mean score, and the Kullback-Leibler divergence
    """

    plot.plot(hist_vi.means(), label="Mean score")
    plot.plot(hist_vi.VI_scores(), label="VI scores")
    plot.plot(hist_vi.VI_scores() - hist_vi.means(), label="Temperature * KL")
    plot.xlabel("Step")
    plot.legend()
    plot.title("Evolution of Variational inference procedure scores")

    return plot
