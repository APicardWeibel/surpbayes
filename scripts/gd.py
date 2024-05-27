# pylint: disable=all
import argparse
import os
import sys
import warnings

import numpy as np
from surpbayes.pyadm1.basic_classes import (
    load_dig_feed,
    load_dig_info,
    load_dig_state,
    load_dig_states,
)
from surpbayes.pyadm1.digester import Digester
from surpbayes.pyadm1.proba import Interface, prior_param, proba_map
from surpbayes.types import ProbaParam

temperature = 0.002
quantiles = [0.1, 0.5, 0.9]
data_path = "ADM1_data_LF"

loc_path, _ = os.path.split(__file__)

data_path = os.path.join(loc_path, "..", "data", "ADM1_data_LF")
save_path = os.path.join(loc_path, "..", "exp_results", "optim", "GD")
os.makedirs(save_path, exist_ok=True)

feed = load_dig_feed(
    os.path.join(data_path, "train_data/feed.csv")
)  # [:100] # limit to first 25 days to speed up
ini_state = load_dig_state(os.path.join(data_path, "train_data/init_state.json"))
obs = load_dig_states(os.path.join(data_path, "train_data/obs.csv"))
dig_info = load_dig_info(os.path.join(data_path, "dig_info.json"))

# Preparing interface to ADM1 model
dig = Digester(dig_info, feed, ini_state, obs)
interface = Interface(dig, temperature=temperature, prior_param=prior_param)

# For correct evaluation
n_eval = 160


def main(per_step: int, eta: float, budget: int = 9600):

    par_str = f"{str(eta).replace('.', '_')}_{per_step}"

    def eval_results(post_params: list[ProbaParam]):
        perfs = np.zeros(len(post_params))

        for i, post_param in enumerate(post_params):
            print(f"Eval {i}")
            samples = proba_map(post_param)(n_eval)
            mean_score = np.mean(interface.mult_score(samples))

            perfs[i] = mean_score + interface.temperature * proba_map.kl(
                post_param, prior_param
            )
        return perfs

    chain_length = budget // per_step
    results = []
    for i in range(20):
        try:
            # Bayes calibration procedure
            opt_res = interface.bayes_calibration(
                optimizer="corr_weights",
                eta=eta,
                chain_length=chain_length,
                per_step=per_step,
                kltol=0.0,
                xtol=0.0,
                k=per_step,
                momentum=0.0,
                refuse_conf=1.0,
                corr_eta=1.0,
            )

            # Save list of posterior
            params = list(opt_res.hist_param)
            params.append(opt_res.opti_param)

            # Re evaluate
            end_param = opt_res.opti_param
            opti_score = np.mean(
                interface.mult_score(proba_map(end_param)(per_step))
            ) + interface.temperature * proba_map.kl(end_param, prior_param)

            print("Not reevaluating, since each estimate is unbiased + independant")
            # Note that when per_step=80, the estimations of the performance are
            # still unbiased/independent, but with larger fluctuations (variance twice
            # as large)
            # This will have some impact in the uncertainty quantification between
            # repeats (increase variance), but no expected to be too significant.

            score_evol = np.array(list(opt_res.hist_score) + [opti_score])
            results.append(score_evol)
            print()

        except Exception as exc:
            # Continue if fails for some reason (proceed to next GD repeat)
            print(f"Failed some reason {exc}")

    # Do computation
    perfs = np.array(results)
    np.savetxt(os.path.join(save_path, f"perfs_{par_str}.csv"), perfs)


if __name__ == "__main__":

    eta_grid = [0.025, 0.05, 0.07]
    per_step_grid = [80, 160]

    # For preliminary tests
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="prelim", type=str, required=False)
    parser.add_argument("-eta", "--eta", default=-1, type=float, required=False)
    parser.add_argument("-ps", "--per_step", default=-1, type=int, required=False)

    args = parser.parse_args()
    if args.mode == "prelim":
        # Correct save_path value
        save_path = os.path.join(save_path, "prelim")
        os.makedirs(save_path)
        print("Perform preliminary assessment of hyperparameters")
        for eta in eta_grid:
            for per_step in per_step_grid:
                main(per_step, eta, 1600)
        sys.exit()
    elif args.mode != "eval":
        warnings.warn(
            "Run mode not correct (should be either 'prelim' or 'eval'). Trying 'eval' run"
        )

    if (args.eta < 0) or (args.per_step < 0):
        raise ValueError(
            "Values for eta and per_step must be specified and positive.",
            "Consider running 'python PATH/TO/FOLD/gd.py -eta <eta> -ps <per_step>'",
        )
    main(args.per_step, args.eta, budget=9600)
