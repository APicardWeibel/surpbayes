# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from surpbayes.pyadm1.basic_classes.cod_vs_conv import COD_VS


class NegativeStates(ValueError):
    """Error for when negative values are passed to states"""


pred_col = [
    "time",
    "S_su",
    "S_aa",
    "S_fa",
    "S_va",
    "S_bu",
    "S_pro",
    "S_ac",
    "S_h2",
    "S_ch4",
    "S_IC",
    "S_IN",
    "S_I",
    "X_c",
    "X_ch",
    "X_pr",
    "X_li",
    "X_su",
    "X_aa",
    "X_fa",
    "X_c4",
    "X_pro",
    "X_ac",
    "X_h2",
    "X_I",
    "S_cation",
    "S_anion",
    "pH",
    "S_va_ion",
    "S_bu_ion",
    "S_pro_ion",
    "S_ac_ion",
    "S_hco3_ion",
    "S_nh3",
    "S_gas_h2",
    "S_gas_ch4",
    "S_gas_co2",
    "S_co2",
    "S_nh4_ion",
    "q_gas",
    "q_ch4",
    "p_ch4",
    "p_co2",
    "VS",
    "VSR",
]

predict_units_dict = {
    "time": "Day",
    "S_su": "kgCOD M-3",
    "S_aa": "kgCOD M-3",
    "S_fa": "kgCOD M-3",
    "S_va": "kgCOD M-3",
    "S_bu": "kgCOD M-3",
    "S_pro": "kgCOD M-3",
    "S_ac": "kgCOD M-3",
    "S_h2": "kgCOD M-3",
    "S_ch4": "kgCOD M-3",
    "S_IC": "kmole C M-3",
    "S_IN": "kmole N M-3",
    "S_I": "kgCOD M-3",
    "X_c": "kgCOD M-3",
    "X_ch": "kgCOD M-3",
    "X_pr": "kgCOD M-3",
    "X_li": "kgCOD M-3",
    "X_su": "kgCOD M-3",
    "X_aa": "kgCOD M-3",
    "X_fa": "kgCOD M-3",
    "X_c4": "kgCOD M-3",
    "X_pro": "kgCOD M-3",
    "X_ac": "kgCOD M-3",
    "X_h2": "kgCOD M-3",
    "X_I": "kgCOD M-3",
    "S_cation": "kmole M-3",
    "S_anion": "kmole M-3",
    "pH": "pH",
    "S_va_ion": "kgCOD M-3",
    "S_bu_ion": "kgCOD M-3",
    "S_pro_ion": "kgCOD M-3",
    "S_ac_ion": "kgCOD M-3",
    "S_hco3_ion": "kmole M-3",
    "S_nh3": "kmole M-3",
    "S_gas_h2": "kgCOD M-3",
    "S_gas_ch4": "kgCOD M-3",
    "S_gas_co2": "kmole M-3",
    "S_co2": "kmole M-3",
    "S_nh4_ion": "kmole M-3",
    "q_gas": "M3 Day-1",
    "q_ch4": "M3 Day-1",
    "p_ch4": "bar",
    "p_co2": "bar",
    "VS": "kgVS M-3",
    "VSR": "ratio",
}

pred_col_dict = {name: i for i, name in enumerate(pred_col)}

cod_vs_dig_states_cols = [pred_col_dict[x] for x in COD_VS.keys()]


class DigesterStates:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @property
    def np_data(self):
        return self._df.to_numpy()

    @property
    def df(self):
        """Panda dataframe representation of the data"""
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame):
        # Check col names
        missing_cols = set(value.columns).difference(pred_col)
        extra_cols = set(pred_col).difference(value.columns)
        if len(extra_cols) > 0:
            raise ValueError(f"Could not interpret the following columns: {extra_cols}")

        if not "time" in value.columns:
            raise ValueError("Observation should have a 'time' column")

        time = value["time"].to_numpy()
        if (len(time) > 1) and np.any(time[1:] - time[:-1] <= 0):
            raise ValueError("Time information should be increasing")

        # Fill missing columns
        if len(missing_cols):
            value[missing_cols] = np.NaN

        if np.any(value.to_numpy() <= 0.0, where=~np.isnan(value.to_numpy())):
            raise NegativeStates("Found negative value")

        self._df = value[pred_col]

    def save(self, path):
        """Save Observations object to a .csv file"""
        self._df.to_csv(path, index=False)
        return path

    def __repr__(self):
        return self._df.__repr__()

    def __str__(self):
        return self._df.__str__()

    def split(self, time_split: float):
        """
        Returns a tuple containing the feed information up to time and the feed information after time.
        Split is done so that the first information can give prediction up to time included.
        """
        time_feed = self._df["time"]
        cond = time_feed < time_split

        obs_before = self._df.iloc[cond]
        obs_after = self._df.iloc[~cond]
        return (DigesterStates(obs_before), DigesterStates(obs_after))

    # def plot(self, pred_name: str, *args, **kwargs):
    #     data = self._df[pred_name]
    #     name_with_unit = f"{pred_name} (in {predict_units_dict[pred_name]})"
    #     if "label" in kwargs.keys():
    #         plt.plot(self._df["time"], data, *args, **kwargs)
    #     else:
    #         plt.plot(self._df["time"], data, *args, label=name_with_unit, **kwargs)
    #     return plt

    def get_state(self, index):
        return DigesterState(self._df.iloc[index])


def load_dig_states(path) -> DigesterStates:
    return DigesterStates(pd.read_csv(path))


ode_state_col = pred_col[1:37]


class DigesterState:
    def __init__(self, state: pd.Series):
        self.df = state.copy()

    @property
    def np_data(self):
        return self._df.to_numpy()

    @property
    def df(self) -> pd.Series:
        """Panda dataframe representation of the data"""
        return self._df

    @df.setter
    def df(self, value: pd.Series):
        # Check if all data is present
        missing = set(pred_col).difference(value.index)
        extra = set(value.index).difference(pred_col)

        err_msg = []
        if missing:
            err_msg.append(f"Missing indexes: {missing}")

        if extra:
            err_msg.append(f"Unknown indexes: {missing}")

        err_str = "\nFurthermore: ".join(err_msg)
        if err_str:
            raise ValueError(err_str)

        self._df = value[pred_col]

    @property
    def t0(self) -> float:
        """Initial date"""
        return self.df["time"]

    @property
    def ode_state(self) -> np.ndarray:
        """State as used for ODE solver in run_adm1"""
        return self.df[ode_state_col].to_numpy()


def load_dig_state(path) -> DigesterState:
    return DigesterState(pd.read_json(path, orient="index", typ="series"))
