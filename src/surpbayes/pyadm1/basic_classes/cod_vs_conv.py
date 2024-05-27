import numpy as np

COD_VS = {
    "S_su": 1.03,
    "S_aa": 1.5,
    "S_fa": 2.0,
    "X_c": 1.42,
    "X_ch": 1.03,
    "X_pr": 1.5,
    "X_li": 2.0,
    "X_su": 1.42,
    "X_aa": 1.42,
    "X_fa": 1.42,
    "X_c4": 1.42,
    "X_pro": 1.42,
    "X_ac": 1.42,
    "X_h2": 1.42,
    "X_I": 1.5,
}  # gCOD / gVS

cod_vs_values = np.array(list(COD_VS.values()))
