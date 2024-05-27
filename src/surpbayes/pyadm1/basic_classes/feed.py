import numpy as np
import pandas as pd
from surpbayes.pyadm1.basic_classes.cod_vs_conv import COD_VS

influent_state_cols = [
    "time",
    "S_su",  # kgCOD M-3
    "S_aa",  # kgCOD M-3
    "S_fa",  # kgCOD M-3
    "S_va",  # kgCOD M-3
    "S_bu",  # kgCOD M-3
    "S_pro",  # kgCOD M-3
    "S_ac",  # kgCOD M-3
    "S_h2",  # kgCOD M-3
    "S_ch4",  # kgCOD M-3
    "S_IC",  # kgCOD M-3
    "S_IN",  # kmole N M-3
    "S_I",  # kmole C M-3
    "X_c",  # kgDCO m-3
    "X_ch",  # kgDCO m-3
    "X_pr",  # kgCOD M-3
    "X_li",  # kgCOD M-3
    "X_su",  # kgCOD M-3
    "X_aa",  # kgCOD M-3
    "X_fa",  # kgCOD M-3
    "X_c4",  # kgCOD M-3
    "X_pro",  # kgCOD M-3
    "X_ac",  # kgCOD M-3
    "X_h2",  # kgCOD M-3
    "X_I",  # kgCOD M-3
    "S_cation",  # kmole M-3
    "S_anion",  # kmole M-3
    "Q",  # M3 Day-1
]

influent_state_col_dict = {name:i for i, name in enumerate(influent_state_cols)}

influent_state_units = {
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
    "Q": "M3 Day-1",
}

# Build time assert: check that all columns have a unit
__delta = set(influent_state_cols).symmetric_difference(influent_state_units)
if len(__delta) > 0:
    raise ValueError(f"Unnexplained columns in Feed: {__delta}")

cod_vs_feed_cols = [influent_state_col_dict[x] for x in COD_VS.keys()]

class Feed:
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
        unexplained = set(value.columns).symmetric_difference(influent_state_cols)
        if len(unexplained) > 0:
            raise ValueError(
                f"Passed dataframe has extra/missing columns: {unexplained}"
            )

        time = value["time"].to_numpy()
        if (len(time) > 1) and np.any(time[1:] - time[:-1] <= 0):
            raise ValueError("Time information should be increasing")
        self._df = value[influent_state_cols] # Reorder

    def save(self, path):
        """Save DigesterFeed object to a .csv file"""
        self._df.to_csv(path, index=False)

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

        feed_before = self._df.iloc[cond]
        feed_after = self._df.iloc[~cond]
        return (Feed(feed_before), Feed(feed_after))

def load_dig_feed(path) -> Feed:
    """
    Loads a digester feed from a csv file.
    """
    return Feed(pd.read_csv(path))
