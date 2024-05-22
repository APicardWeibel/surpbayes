# pylint: disable=all
import os

import numpy as np
from surpbayes.pyadm1.basic_classes import (load_dig_feed, load_dig_info,
                                            load_dig_state, load_dig_states)
from surpbayes.pyadm1.digester import Digester
from surpbayes.pyadm1.proba import Interface, prior_param, proba_map
from surpbayes.types import ProbaParam

# Set parameters
temperature = 0.002
quantiles = [0.2, 0.5, 0.8]

# Number of draws for independant evaluation
n_eval = 160

# Accessing data
loc_path, _ = os.path.split(__file__)

data_path = os.path.join(loc_path, "..", "ADM1_data_LF")
save_path = os.path.join(loc_path, "..", "exp_results", "optim")

os.makedirs(save_path, exist_ok=True)

feed = load_dig_feed(os.path.join(data_path,"train_data/feed.csv")) 
ini_state = load_dig_state(os.path.join(data_path, "train_data/init_state.json"))
obs = load_dig_states(os.path.join(data_path, "train_data/obs.csv"))
dig_info = load_dig_info(os.path.join(data_path, "dig_info.json"))

# Preparing interface to ADM1 model
dig = Digester(dig_info, feed, ini_state, obs)
interface = Interface(dig, temperature, prior_param=prior_param)


def eval_results(
    post_params:list[ProbaParam]
):
    """Evaluate a posterior"""
    perfs = np.zeros(len(post_params))

    for i, post_param in enumerate(post_params):
        print(f"Eval {i}")
        samples = proba_map(post_param)(n_eval)
        mean_score = np.mean(interface.mult_score(samples))

        perfs[i] = mean_score + interface.temperature * proba_map.kl(post_param, prior_param)
    return perfs


# Main call 
results = []
for i in range(20):
    per_step = [160] + [32] * 295
    opt_res = interface.bayes_calibration(
        optimizer="score_approx",
        n_estim_weights=4*10**4,
        kl_max=1.0,
        dampen=0.5,
        chain_length=len(per_step),
        per_step=per_step,
        kltol=0.0, # Force complete
        m_max=25,
        )
    params = np.asarray(opt_res.hist_param)
    np.savetxt(os.path.join(save_path, f"hist_par_{i}.csv"), 
               params)
    opt_res.save(f"opt_res_{i}", path=save_path)

    print("Re evaluating score evolution (this can be long)")
    # Speed up by evaluating only part of the indexes
    evals_index = list(range(5)) + list(range(5, 30, 5)) + list(range(50, 100, 10)) + list(range(100, 300, 25)) + [-1]
    params_to_eval = [opt_res.hist_param[i] for i in evals_index]

    score_evol = eval_results(params_to_eval)
    np.savetxt(os.path.join(save_path, f"score_evol_{i}"), score_evol)
    results.append(score_evol)
    print()

# Do computation
perfs = np.array(results)
np.savetxt(os.path.join(save_path, "perfs_sa.csv"), perfs)
