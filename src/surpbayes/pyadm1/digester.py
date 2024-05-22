import warnings
from typing import Optional

import pandas as pd
from surpbayes.pyadm1.basic_classes.dig_info import DigesterInformation
from surpbayes.pyadm1.basic_classes.feed import Feed
from surpbayes.pyadm1.basic_classes.obs import (DigesterState, DigesterStates,
                                           NegativeStates, pred_col)
from surpbayes.pyadm1.basic_classes.parameter import ADM1Param
from surpbayes.pyadm1.model.run_adm1 import run_adm1
from surpbayes.pyadm1.prediction_error import adm1_err


class ADM1Failure(Warning):
    """Warning class when ADM1 computations failed"""

pred_names = [
    "S_va",
    "S_bu",
    "S_pro",
    "S_ac",
    "S_IN",
    "q_gas",
    "q_ch4",
    "p_ch4",
    "p_co2",
]

class Digester:
    def __init__(
        self,
        dig_info: DigesterInformation,
        feed: Feed,
        ini_state: DigesterState,
        obs: Optional[DigesterStates],
    ):
        self.dig_info = dig_info
        self.feed = feed
        self.ini_state = ini_state
        self.obs = obs


    @property
    def dig_info(self):
        return self._dig_info

    @dig_info.setter
    def dig_info(self, value):
        if not isinstance(value, DigesterInformation):
            raise TypeError("dig_info must be a DigesterInformation")
        self._dig_info = value

    @property
    def feed(self):
        return self._feed

    @feed.setter
    def feed(self, value):
        if not isinstance(value, Feed):
            raise TypeError("feed must be a Feed")
        self._feed = value
        self._np_feed = self._feed.np_data

    @property
    def ini_state(self):
        return self._ini_state

    @ini_state.setter
    def ini_state(self, value):
        if not isinstance(value, DigesterState):
            raise TypeError("ini_state must be a DigesterState")
        self._ini_state = value
        self._np_ini_state = self._ini_state.np_data

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, value):
        if value is None:
            self._obs = None
            self._np_obs = None
            return None
        if not isinstance(value, DigesterStates):
            raise TypeError("obs must be a DigesterStates")
        self._obs = value
        self._np_obs = self._obs.np_data

    def simulate(
        self,
        param: ADM1Param,
        solver_method: str = "LSODA",
        max_step: float = 60.0 / (24.0 * 60.0),
        min_step: float = 10**-6,
        **kwargs
    ) -> DigesterStates:

        out =  run_adm1(
            influent_state=self._np_feed,
            initial_state=self._np_ini_state,
            V_liq=self._dig_info.V_liq,
            V_gas=self._dig_info.V_gas,
            T_ad=self._dig_info.T_ad,
            T_op=self._dig_info.T_op,
            **param.param,
            solver_method=solver_method,
            max_step=max_step,
            min_step=min_step,
            **kwargs
        )
        return DigesterStates(pd.DataFrame(out, columns = pred_col))

    def score(
        self,
        param: ADM1Param,
        solver_method :str = "LSODA",
        max_step: float = 60.0 / (24.0 * 60.0),
        min_step: float = 10**-6,
        # Score arguments
        eps:float = 10** -8,
        max_score: float = 3.0,
        elbow: float = 2.0,
        silent: bool = True,
        **kwargs
    ):
        try:
            pred = self.simulate(
                param=param,
                solver_method=solver_method,
                max_step=max_step,
                min_step=min_step,
                **kwargs
            )

            return adm1_err(pred = pred, obs = self.obs, eps=eps, max_score=max_score, elbow=elbow)
        except (RuntimeWarning, UserWarning, NegativeStates, ZeroDivisionError) as exc:
            if not silent:
                warnings.warn(
                    f"Could not compute error for parameter:\n{param}\n\n{exc}\n",
                    category=ADM1Failure,
                )
            return max_score

