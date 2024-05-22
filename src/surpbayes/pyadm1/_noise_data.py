"""
Noise Data

Routines used to noise data for Calibration/UQ benchmark
"""
from copy import copy
import warnings

import numpy as np

from surpbayes.pyadm1.basic_classes import Feed, DigesterState, DigesterStates


def noise_influent(influent_state: Feed, noise_lev: float) -> Feed:
    """
    Noise influent (except time) with log-uniform multiplicative factor.
    No side effect on input.

    Arguments:
        influent_state: DigesterFeed to noise
        noise_lev: noise level used
    """

    inf_state_loc = copy(influent_state)
    if noise_lev > 0:
        noise_lev = -noise_lev
        noise = np.reshape(
            np.exp(
                np.random.uniform(
                    (-noise_lev),
                    noise_lev,
                    influent_state.df.shape[0] * (influent_state.df.shape[1] - 1),
                )
            ),
            (influent_state.df.shape[0], influent_state.df.shape[1] - 1),
        )
        inf_state_loc.df.iloc[:, 1:] = inf_state_loc.df.iloc[:, 1:] * noise
    elif noise_lev < 0:
        warnings.warn(
            "noise level given is negative. Returning influent_state unchanged"
        )
    return inf_state_loc


def noise_obs(obs: DigesterStates, noise_lev: float) -> DigesterStates:
    """
    Noise DigesterStates object (except time) with log-uniform multiplicative factor.
    No side effect on obs.

    Arguments:
        obs: DigesterStates to noise
        noise_lev: noise level used
    """
    obs_loc = copy(obs)
    if noise_lev > 0:
        noise = np.reshape(
            np.exp(
                np.random.uniform(
                    (-noise_lev), noise_lev, obs.df.shape[0] * (obs.df.shape[1] - 1)
                )
            ),
            (obs.df.shape[0], obs.df.shape[1] - 1),
        )
        obs_loc.df.iloc[:, 1:] = obs_loc[:, 1:] * noise
    elif noise_lev < 0:
        warnings.warn("noise level given is negative. Returning obs unchanged")
    return obs_loc


def noise_init_state(init_state: DigesterState, noise_lev: float) -> DigesterState:
    """
    Noise DigesterState object (except time) with log-uniform multiplicative factor.
    No side effect on init_state.

    Arguments:
        init_state: DigesterState to noise
        noise_lev: noise level used
    """
    init_loc = copy(init_state)
    if noise_lev > 0:
        noise = np.exp(np.random.uniform((-noise_lev), noise_lev, len(init_loc) - 1))
        init_loc.df.iloc[1:] = init_loc.df.iloc[1:] * noise
    elif noise_lev < 0:
        warnings.warn("noise level given is negative. Returning init_state unchanged")
    return init_loc
