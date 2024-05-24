import os

import matplotlib.pyplot as plt
from surpbayes.bayes.optim_result_bayes import OptimResultBayes
from surpbayes.bayes.plot.accu import plot_scores
from surpbayes.bayes.plot.hist_bayes import plot_hist_bayes
from surpbayes.bayes.plot.optim_result import plot_score_evol


def make_analysis_plots(optim_result: OptimResultBayes, save_path="."):
    plot = plot_scores(optim_result.sample_val, plot=plt)
    plot.savefig(os.path.join(save_path, "scores_scatter.pdf"))
    plot.show()
    plot.clf()

    plot = plot_hist_bayes(optim_result.log_bayes, plot=plt)
    plot.savefig(os.path.join(save_path, "KPI_evol.pdf"))
    plot.show()
    plot.clf()

    plot = plot_score_evol(optim_result, plot=plt)
    plot.savefig(os.path.join(save_path, "scores_evol.pdf"))
    plot.show()
    plot.clf()
