# pylint: disable=all
import argparse

import pandas as pd
from azureml.core import Run
from codai_machinery.arguments_parser import parse_and_log_script_arguments
import os

import numpy as np
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.pyadm1.basic_classes import (load_dig_feed, load_dig_info,
                                       load_dig_state, load_dig_states)
from surpbayes.pyadm1.digester import Digester
from surpbayes.pyadm1.proba import Interface, prior_param, proba_map
from surpbayes.types import ProbaParam

run = Run.get_context()
parser = argparse.ArgumentParser()
args = parse_and_log_script_arguments(parser, run)

temperature = 0.002
quantiles = [0.1, 0.5, 0.9]
data_path = "ADM1_data_LF"

save_path = "outputs/"
os.makedirs(save_path, exist_ok=True)

feed = load_dig_feed(os.path.join(data_path,"train_data/feed.csv"))# [:100] # limit to first 25 days to speed up
ini_state = load_dig_state(os.path.join(data_path, "train_data/init_state.json"))
obs = load_dig_states(os.path.join(data_path, "train_data/obs.csv"))
dig_info = load_dig_info(os.path.join(data_path, "dig_info.json"))

dig = Digester(dig_info, feed, ini_state, obs)

interface = Interface(dig, temperature, prior_param=prior_param)

# For correct evaluation
n_eval = 160
# def eval_results(
#     post_params:list[ProbaParam],
#     accu:AccuSampleVal
# ):
#     gens = accu.gen_tracker()
#     perfs = np.zeros(len(post_params))
#     f_gen = gens[0]

#     for i, post_param in enumerate(post_params):
#         prev_score = accu.vals()[gens == (f_gen - i)]
#         n_prev_eval = len(prev_score)

#         if n_prev_eval < n_eval:  
#             samples = proba_map(post_param)(n_eval- n_prev_eval)
#             scores = interface.mult_score(samples)
#             mean_score = (np.sum(prev_score) + np.sum(scores))/ n_eval
#         else:
#             mean_score = np.mean(prev_score)

#         perfs[i] = mean_score + interface.temperature * proba_map.kl(post_param, prior_param)
#     return perfs

def eval_results(
    post_params:list[ProbaParam]
):
    perfs = np.zeros(len(post_params))

    for i, post_param in enumerate(post_params):
        print(f"Eval {i}")
        samples = proba_map(post_param)(n_eval)
        mean_score = np.mean(interface.mult_score(samples))

        perfs[i] = mean_score + interface.temperature * proba_map.kl(post_param, prior_param)
    return perfs

results = []
for i in range(20):
    try:
        per_step = int(args.per_step)
        chain_length = (32 * 300) //per_step
        opt_res = interface.bayes_calibration(
            optimizer="corr_weights",
            eta=float(args.eta), # Calibrate this term or else!
            chain_length=chain_length,
            per_step=per_step,
            kltol=0.0,
            xtol=0.0,
            k=per_step, # No weight correction active
            momentum=0.0,
            refuse_conf=1.0,
            corr_eta=1.0,
        )

        params = list(opt_res.hist_param)
        params.append(opt_res.opti_param)
        np.savetxt(os.path.join(save_path, f"hist_par_{i}.csv"), 
                np.array(params))
        opt_res.save(f"opt_res_{i}", path=save_path)

        end_param = opt_res.opti_param
        opti_score = np.mean(interface.mult_score(proba_map(end_param)(per_step))) + interface.temperature * proba_map.kl(end_param, prior_param)

        print("Not reevaluating, since each estimate is unbiased + independant")
        score_evol = np.array(list(opt_res.hist_score) + [opti_score])
        np.savetxt(os.path.join(save_path, f"score_evol_{i}"), score_evol)
        results.append(score_evol)
        print()
    except Exception as exc:
        print(f"Failed some reason {exc}")

# Do computation
perfs = np.array(results)
np.savetxt(os.path.join(save_path, "perfs.csv"), perfs)

quant = np.apply_along_axis(lambda x: np.quantile(x, quantiles), axis=0, arr=perfs)
run.complete()

