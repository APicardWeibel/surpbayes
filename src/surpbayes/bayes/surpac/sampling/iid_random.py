"""
Baseline for generation of new samples to evaluate
"""
from typing import Callable

from surpbayes.accu_xy import AccuSampleVal
from surpbayes.misc import par_eval
from surpbayes.proba import Proba


def enrich_sample_iid(
    proba: Proba,
    accu_sample: AccuSampleVal,
    score: Callable,
    n_new: int,
    vectorized: bool = False,
    parallel: bool = True,
):
    """Enriching samples using SVGD algorithm"""

    new_points = proba(n_new)

    if vectorized:
        values = score(new_points)
    else:
        values = par_eval(score, new_points, parallel)

    accu_sample.add(new_points, values)  # in place
