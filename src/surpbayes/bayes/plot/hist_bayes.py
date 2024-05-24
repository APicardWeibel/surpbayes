import matplotlib.pyplot as plt
from surpbayes.bayes.hist_bayes import HistBayesLog


def plot_hist_bayes(hist_bayes: HistBayesLog, plot=plt):
    """
    Plot evolution of KPIs of a PAC-Bayes objective optimisation task.
    KPIs plotted are the PAC-Bayes objective, the mean score, and the Kullback-Leibler
    divergence
    """

    plot.plot(hist_bayes.means(), label="Mean score")
    plot.plot(hist_bayes.bayes_scores(), label="PAC-Bayes obj.")
    plot.plot(hist_bayes.bayes_scores() - hist_bayes.means(), label="Temperature * KL")
    plot.xlabel("Step")
    plot.legend()
    plot.title("Evolution of PAC-Bayes onjective")

    return plot
