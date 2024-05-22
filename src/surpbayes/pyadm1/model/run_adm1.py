"""
Modelisation of Anaerobic Digestion through ADM1.

Implementation based on PyADM1 package https://github.com/CaptainFerMag/PyADM1.

Main change:
- the implementation is now a function and not a script.
Further change:
- Notation X_xc is changed to X_c, coherent with ADM1 formulation in
https://doi.org/10.2166/wst.2002.0292
- Output accus are prepared in advance with full size rather than iteratively appended. This
appeared to decrease computation time significantly.
- The use of global variables is removed. Instead, the ODE function is defined inside the main
function.
- VS/VSR computations added.

Note:
    Function signature uses the custom classes defined in the pyadm1 module. The function would
    perfom in a similar manner if numpy.ndarrays correctly formatted are used instead.
"""

import warnings
from functools import partial
from math import exp

import numba as nb
import numpy as np
import scipy.integrate
import surpbayes.nb_typ as nbt
from surpbayes.pyadm1.basic_classes.Phy import R, T_base, p_atm
from surpbayes.pyadm1.model.vs_computation import dig_states_vs, feed_vs


## ----------- Specify constant values -----------
class InstabilityError(Exception):
    """Custom class for absurd values during AD simulation"""


@nb.njit(
    nbt.f1D(
        nbt.f,  # t: float,
        nbt.f1D,  # state_zero: np.ndarray,
        nbt.f,  # K_pH_aa: float,
        nbt.f,  # nn_aa: float,
        nbt.f,  # K_pH_h2: float,
        nbt.f,  # n_h2: float,
        nbt.f,  # K_pH_ac:float,
        nbt.f,  # n_ac:float,
        nbt.f,  # K_S_IN,
        nbt.f,  # K_I_h2_fa,
        nbt.f,  # K_H_h2,
        nbt.f,  # K_I_h2_c4,
        nbt.f,  # K_I_h2_pro,
        nbt.f,  # K_I_nh3,
        nbt.f,  # k_dis,
        nbt.f,  # k_hyd_ch,
        nbt.f,  # k_hyd_pr,
        nbt.f,  # k_hyd_li,
        nbt.f,  # k_m_su,
        nbt.f,  # K_S_su,
        nbt.f,  # k_m_aa,
        nbt.f,  # K_S_aa,
        nbt.f,  # k_m_pro: float,
        nbt.f,  # K_S_pro: float,
        nbt.f,  # k_m_fa: float,
        nbt.f,  # K_S_fa: float,
        nbt.f,  # k_m_c4,
        nbt.f,  # K_S_c4,
        nbt.f,  # k_m_ac,
        nbt.f,  # K_S_ac,
        nbt.f,  # k_dec: float,
        nbt.f,  # k_m_h2: float,
        nbt.f,  # K_S_h2: float,
        nbt.f,  # T_op: float,
        nbt.f,  # p_gas_h2o,
        nbt.f,  # V_liq: float,
        nbt.f,  # q_ad,
        nbt.f,  # K_H_ch4,
        nbt.f,  # K_H_co2,
        nbt.f,  # S_su_in,
        nbt.f,  # S_aa_in,
        nbt.f,  # S_fa_in,
        nbt.f,  # S_va_in,
        nbt.f,  # S_bu_in,
        nbt.f,  # S_pro_in,
        nbt.f,  # S_ac_in,
        nbt.f,  # S_ch4_in,
        nbt.f,  # s_1,
        nbt.f,  # s_2,
        nbt.f,  # s_3,
        nbt.f,  # s_4,
        nbt.f,  # s_5,
        nbt.f,  # s_6,
        nbt.f,  # s_7,
        nbt.f,  # s_8,
        nbt.f,  # s_9,
        nbt.f,  # s_10,
        nbt.f,  # s_11,
        nbt.f,  # s_12,
        nbt.f,  # s_13,
        nbt.f,  # V_gas,
        nbt.f,  # S_IC_in,
        nbt.f,  # S_IN_in,
        nbt.f,  # S_I_in,
        nbt.f,  # X_c_in,
        nbt.f,  # X_ch_in,
        nbt.f,  # X_pr_in,
        nbt.f,  # X_li_in,
        nbt.f,  # X_su_in,
        nbt.f,  # X_aa_in,
        nbt.f,  # X_fa_in,
        nbt.f,  # X_c4_in,
        nbt.f,  # X_pro_in,
        nbt.f,  # X_ac_in,
        nbt.f,  # X_h2_in,
        nbt.f,  # X_I_in,
        nbt.f,  # S_cation_in,
        nbt.f,  # S_anion_in,
        nbt.f,  # N_xc
        nbt.f,  # N_I
        nbt.f,  # N_aa
        nbt.f,  # N_bac
        nbt.f,  # f_sI_xc
        nbt.f,  # f_xI_xc
        nbt.f,  # f_ch_xc
        nbt.f,  # f_pr_xc
        nbt.f,  # f_li_xc
        nbt.f,  # f_fa_li
        nbt.f,  # f_bu_su
        nbt.f,  # f_pro_su
        nbt.f,  # f_ac_su
        nbt.f,  # f_va_aa
        nbt.f,  # f_bu_aa
        nbt.f,  # f_pro_aa
        nbt.f,  # f_ac_aa
        nbt.f,  # Y_su
        nbt.f,  # Y_aa
        nbt.f,  # Y_fa
        nbt.f,  # Y_c4
        nbt.f,  # Y_pro
        nbt.f,  # Y_ac
        nbt.f,  # Y_h2
        nbt.f,  # k_p
        nbt.f,  # k_L_a
    )
)
def adm1_ode(
    t: float,
    state_zero: np.ndarray,
    K_pH_aa: float,
    nn_aa: float,
    K_pH_h2: float,
    n_h2: float,
    K_pH_ac: float,
    n_ac: float,
    K_S_IN: float,
    K_I_h2_fa: float,
    K_H_h2: float,
    K_I_h2_c4: float,
    K_I_h2_pro: float,
    K_I_nh3: float,
    k_dis: float,
    k_hyd_ch: float,
    k_hyd_pr: float,
    k_hyd_li: float,
    k_m_su: float,
    K_S_su: float,
    k_m_aa: float,
    K_S_aa: float,
    k_m_pro: float,
    K_S_pro: float,
    k_m_fa: float,
    K_S_fa: float,
    k_m_c4: float,
    K_S_c4: float,
    k_m_ac: float,
    K_S_ac: float,
    k_dec: float,
    k_m_h2: float,
    K_S_h2: float,
    T_op: float,
    p_gas_h2o: float,
    V_liq: float,
    q_ad: float,
    K_H_ch4: float,
    K_H_co2: float,
    S_su_in: float,
    S_aa_in: float,
    S_fa_in: float,
    S_va_in: float,
    S_bu_in: float,
    S_pro_in: float,
    S_ac_in: float,
    S_ch4_in: float,
    s_1: float,
    s_2: float,
    s_3: float,
    s_4: float,
    s_5: float,
    s_6: float,
    s_7: float,
    s_8: float,
    s_9: float,
    s_10: float,
    s_11: float,
    s_12: float,
    s_13: float,
    V_gas: float,
    S_IC_in: float,
    S_IN_in: float,
    S_I_in: float,
    X_c_in: float,
    X_ch_in: float,
    X_pr_in: float,
    X_li_in: float,
    X_su_in: float,
    X_aa_in: float,
    X_fa_in: float,
    X_c4_in: float,
    X_pro_in: float,
    X_ac_in: float,
    X_h2_in: float,
    X_I_in: float,
    S_cation_in: float,
    S_anion_in: float,
    # New parameters
    N_xc: float,
    N_I: float,
    N_aa: float,
    N_bac: float,
    f_sI_xc: float,
    f_xI_xc: float,
    f_ch_xc: float,
    f_pr_xc: float,
    f_li_xc: float,
    f_fa_li: float,
    f_bu_su: float,
    f_pro_su: float,
    f_ac_su: float,
    f_va_aa: float,
    f_bu_aa: float,
    f_pro_aa: float,
    f_ac_aa: float,
    Y_su: float,
    Y_aa: float,
    Y_fa: float,
    Y_c4: float,
    Y_pro: float,
    Y_ac: float,
    Y_h2: float,
    k_p: float,
    k_L_a: float,
):
    # pylint: disable=W0613
    ## Unpack state_zero
    S_su = state_zero[0]
    S_aa = state_zero[1]
    S_fa = state_zero[2]
    S_va = state_zero[3]
    S_bu = state_zero[4]
    S_pro = state_zero[5]
    S_ac = state_zero[6]
    S_h2 = state_zero[7]
    S_ch4 = state_zero[8]
    S_IC = state_zero[9]
    S_IN = state_zero[10]
    S_I = state_zero[11]
    X_c = state_zero[12]
    X_ch = state_zero[13]
    X_pr = state_zero[14]
    X_li = state_zero[15]
    X_su = state_zero[16]
    X_aa = state_zero[17]
    X_fa = state_zero[18]
    X_c4 = state_zero[19]
    X_pro = state_zero[20]
    X_ac = state_zero[21]
    X_h2 = state_zero[22]
    X_I = state_zero[23]
    S_cation = state_zero[24]
    S_anion = state_zero[25]
    S_H_ion = state_zero[26]
    S_hco3_ion = state_zero[31]
    S_nh3 = state_zero[32]
    S_gas_h2 = state_zero[33]
    S_gas_ch4 = state_zero[34]
    S_gas_co2 = state_zero[35]

    # Main part of the function

    I_pH_aa = (K_pH_aa**nn_aa) / (S_H_ion**nn_aa + K_pH_aa**nn_aa)
    I_pH_h2 = (K_pH_h2**n_h2) / (S_H_ion**n_h2 + K_pH_h2**n_h2)

    I_pH_ac = (K_pH_ac**n_ac) / (S_H_ion**n_ac + K_pH_ac**n_ac)
    I_IN_lim = S_IN / (S_IN + K_S_IN)
    I_h2_fa = K_I_h2_fa / (K_I_h2_fa + S_h2)
    I_h2_c4 = K_I_h2_c4 / (K_I_h2_c4 + S_h2)
    I_h2_pro = K_I_h2_pro / (K_I_h2_pro + S_h2)
    I_nh3 = K_I_nh3 / (K_I_nh3 + S_nh3)

    I_5 = I_pH_aa * I_IN_lim
    I_6 = I_5
    I_7 = I_pH_aa * I_IN_lim * I_h2_fa
    I_8 = I_pH_aa * I_IN_lim * I_h2_c4
    I_9 = I_8
    I_10 = I_pH_aa * I_IN_lim * I_h2_pro
    I_11 = I_pH_ac * I_IN_lim * I_nh3
    I_12 = I_pH_h2 * I_IN_lim

    # biochemical process rates from Rosen et al (2006) BSM2 report
    Rho_1 = k_dis * X_c  # Disintegration
    Rho_2 = k_hyd_ch * X_ch  # Hydrolysis of carbohydrates
    Rho_3 = k_hyd_pr * X_pr  # Hydrolysis of proteins
    Rho_4 = k_hyd_li * X_li  # Hydrolysis of lipids
    Rho_5 = k_m_su * (S_su / (K_S_su + S_su)) * X_su * I_5  # Uptake of sugars
    Rho_6 = k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6  # Uptake of amino-acids
    Rho_7 = (
        k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7
    )  # Uptake of LCFA (long-chain fatty acids)
    Rho_8 = (
        k_m_c4 * (S_va / (K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8
    )  # Uptake of valerate
    Rho_9 = (
        k_m_c4 * (S_bu / (K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9
    )  # Uptake of butyrate
    Rho_10 = (
        k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10
    )  # Uptake of propionate
    Rho_11 = k_m_ac * (S_ac / (K_S_ac + S_ac)) * X_ac * I_11  # Uptake of acetate
    Rho_12 = k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12  # Uptake of hydrogen
    Rho_13 = k_dec * X_su  # Decay of X_su
    Rho_14 = k_dec * X_aa  # Decay of X_aa
    Rho_15 = k_dec * X_fa  # Decay of X_fa
    Rho_16 = k_dec * X_c4  # Decay of X_c4
    Rho_17 = k_dec * X_pro  # Decay of X_pro
    Rho_18 = k_dec * X_ac  # Decay of X_ac
    Rho_19 = k_dec * X_h2  # Decay of X_h2

    # gas phase algebraic equations from Rosen et al (2006) BSM2 report
    R_T_op = R * T_op
    p_gas_h2 = S_gas_h2 * R_T_op / 16.0
    p_gas_ch4 = S_gas_ch4 * R_T_op / 64.0
    p_gas_co2 = S_gas_co2 * R_T_op

    p_gas = (
        p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o
    )  ## p_gas_h2o is a function of T_ad and T_base -> does not change

    q_gas = k_p * (p_gas - p_atm)
    if q_gas < 0.0:
        q_gas = 0.0

    # gas transfer rates from Rosen et al (2006) BSM2 report
    Rho_T_8 = k_L_a * (S_h2 - 16.0 * K_H_h2 * p_gas_h2)
    Rho_T_9 = k_L_a * (S_ch4 - 64.0 * K_H_ch4 * p_gas_ch4)
    Rho_T_10 = k_L_a * (S_IC - S_hco3_ion - K_H_co2 * p_gas_co2)

    # differential equaitons from Rosen et al (2006) BSM2 report
    # differential equations 1 to 12 (soluble matter)
    diff_S_su = (
        q_ad / V_liq * (S_su_in - S_su) + Rho_2 + (1.0 - f_fa_li) * Rho_4 - Rho_5
    )  # eq1

    diff_S_aa = q_ad / V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6  # eq2

    diff_S_fa = q_ad / V_liq * (S_fa_in - S_fa) + (f_fa_li * Rho_4) - Rho_7  # eq3

    diff_S_va = (
        q_ad / V_liq * (S_va_in - S_va) + (1.0 - Y_aa) * f_va_aa * Rho_6 - Rho_8
    )  # eq4

    diff_S_bu = (
        q_ad / V_liq * (S_bu_in - S_bu)
        + (1.0 - Y_su) * f_bu_su * Rho_5
        + (1.0 - Y_aa) * f_bu_aa * Rho_6
        - Rho_9
    )  # eq5

    diff_S_pro = (
        q_ad / V_liq * (S_pro_in - S_pro)
        + (1.0 - Y_su) * f_pro_su * Rho_5
        + (1.0 - Y_aa) * f_pro_aa * Rho_6
        + (1.0 - Y_c4) * 0.54 * Rho_8
        - Rho_10
    )  # eq6

    diff_S_ac = (
        q_ad / V_liq * (S_ac_in - S_ac)
        + (1.0 - Y_su) * f_ac_su * Rho_5
        + (1.0 - Y_aa) * f_ac_aa * Rho_6
        + (1.0 - Y_fa) * 0.7 * Rho_7
        + (1.0 - Y_c4) * 0.31 * Rho_8
        + (1.0 - Y_c4) * 0.8 * Rho_9
        + (1.0 - Y_pro) * 0.57 * Rho_10
        - Rho_11
    )  # eq7

    # diff_S_h2 is defined with DAE parralel equations

    diff_S_ch4 = (
        q_ad / V_liq * (S_ch4_in - S_ch4)
        + (1.0 - Y_ac) * Rho_11
        + (1.0 - Y_h2) * Rho_12
        - Rho_T_9
    )  # eq9

    ## eq10 start##
    Sigma = (
        s_1 * Rho_1
        + s_2 * Rho_2
        + s_3 * Rho_3
        + s_4 * Rho_4
        + s_5 * Rho_5
        + s_6 * Rho_6
        + s_7 * Rho_7
        + s_8 * Rho_8
        + s_9 * Rho_9
        + s_10 * Rho_10
        + s_11 * Rho_11
        + s_12 * Rho_12
        + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
    )

    ratio_q_ad_V_liq = q_ad / V_liq

    diff_S_IC = ratio_q_ad_V_liq * (S_IC_in - S_IC) - Sigma - Rho_T_10
    ## eq10 end##

    diff_S_IN = (
        ratio_q_ad_V_liq * (S_IN_in - S_IN)
        + (N_xc - f_xI_xc * N_I - f_sI_xc * N_I - f_pr_xc * N_aa) * Rho_1
        - Y_su * N_bac * Rho_5
        + (N_aa - Y_aa * N_bac) * Rho_6
        - Y_fa * N_bac * Rho_7
        - Y_c4 * N_bac * Rho_8
        - Y_c4 * N_bac * Rho_9
        - Y_pro * N_bac * Rho_10
        - Y_ac * N_bac * Rho_11
        - Y_h2 * N_bac * Rho_12
        + (N_bac - N_xc)
        * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
    )  # eq11

    diff_S_I = ratio_q_ad_V_liq * (S_I_in - S_I) + f_sI_xc * Rho_1  # eq12

    # Differential equations 13 to 24 (particulate matter)
    diff_X_c = (
        ratio_q_ad_V_liq * (X_c_in - X_c)
        - Rho_1
        + Rho_13
        + Rho_14
        + Rho_15
        + Rho_16
        + Rho_17
        + Rho_18
        + Rho_19
    )  # eq13

    diff_X_ch = ratio_q_ad_V_liq * (X_ch_in - X_ch) + f_ch_xc * Rho_1 - Rho_2  # eq14

    diff_X_pr = ratio_q_ad_V_liq * (X_pr_in - X_pr) + f_pr_xc * Rho_1 - Rho_3  # eq15

    diff_X_li = ratio_q_ad_V_liq * (X_li_in - X_li) + f_li_xc * Rho_1 - Rho_4  # eq16

    diff_X_su = ratio_q_ad_V_liq * (X_su_in - X_su) + Y_su * Rho_5 - Rho_13  # eq17

    diff_X_aa = ratio_q_ad_V_liq * (X_aa_in - X_aa) + Y_aa * Rho_6 - Rho_14  # eq18

    diff_X_fa = ratio_q_ad_V_liq * (X_fa_in - X_fa) + Y_fa * Rho_7 - Rho_15  # eq19

    diff_X_c4 = (
        ratio_q_ad_V_liq * (X_c4_in - X_c4) + Y_c4 * Rho_8 + Y_c4 * Rho_9 - Rho_16
    )  # eq20

    diff_X_pro = ratio_q_ad_V_liq * (X_pro_in - X_pro) + Y_pro * Rho_10 - Rho_17  # eq21

    diff_X_ac = ratio_q_ad_V_liq * (X_ac_in - X_ac) + Y_ac * Rho_11 - Rho_18  # eq22

    diff_X_h2 = ratio_q_ad_V_liq * (X_h2_in - X_h2) + Y_h2 * Rho_12 - Rho_19  # eq23

    diff_X_I = ratio_q_ad_V_liq * (X_I_in - X_I) + f_xI_xc * Rho_1  # eq24

    # Differential equations 25 and 26 (cations and anions)
    diff_S_cation = ratio_q_ad_V_liq * (S_cation_in - S_cation)  # eq25

    diff_S_anion = ratio_q_ad_V_liq * (S_anion_in - S_anion)  # eq26

    diff_S_h2 = 0

    # Differential equations 27 to 32 (ion states, only for ODE implementation)
    diff_S_va_ion = 0  # eq27 ## Changed with DAESolve
    diff_S_bu_ion = 0  # eq28
    diff_S_pro_ion = 0  # eq29
    diff_S_ac_ion = 0  # eq30
    diff_S_hco3_ion = 0  # eq31
    diff_S_nh3 = 0  # eq32

    # Gas phase equations: Differential equations 33 to 35
    # try:
    diff_S_gas_h2 = (q_gas / V_gas * -1.0 * S_gas_h2) + (
        Rho_T_8 * V_liq / V_gas
    )  # eq33
    # except RuntimeWarning as exc:
    #     print(
    #         f"RuntimeWarning: q_gas: {q_gas}, V_gas: {V_gas}, S_gas_h2: {S_gas_h2}, Rho_T_8: {Rho_T_8}"
    #     )
    #     raise InstabilityError(
    #         "run_adm1: diff_S_gas_h2 can not be defined"
    #     ) from exc

    diff_S_gas_ch4 = (q_gas / V_gas * -1.0 * S_gas_ch4) + (
        Rho_T_9 * V_liq / V_gas
    )  # eq34

    diff_S_gas_co2 = (q_gas / V_gas * -1.0 * S_gas_co2) + (
        Rho_T_10 * V_liq / V_gas
    )  # eq35

    diff_S_H_ion = 0.0

    return np.array(
        [
            diff_S_su,
            diff_S_aa,
            diff_S_fa,
            diff_S_va,
            diff_S_bu,
            diff_S_pro,
            diff_S_ac,
            diff_S_h2,
            diff_S_ch4,
            diff_S_IC,
            diff_S_IN,
            diff_S_I,
            diff_X_c,
            diff_X_ch,
            diff_X_pr,
            diff_X_li,
            diff_X_su,
            diff_X_aa,
            diff_X_fa,
            diff_X_c4,
            diff_X_pro,
            diff_X_ac,
            diff_X_h2,
            diff_X_I,
            diff_S_cation,
            diff_S_anion,
            diff_S_H_ion,
            diff_S_va_ion,
            diff_S_bu_ion,
            diff_S_pro_ion,
            diff_S_ac_ion,
            diff_S_hco3_ion,
            diff_S_nh3,
            diff_S_gas_h2,
            diff_S_gas_ch4,
            diff_S_gas_co2,
        ]
    )


@nb.njit(
    nb.types.Tuple(
        (
            nbt.f1D,  # state_zero,
            nbt.f,  # q_gas_accu,
            nbt.f,  # q_ch4_accu,
            nbt.f,  # p_co2_day_mean_accu,
            nbt.f,  # p_ch4_day_mean_accu
        )
    )(
        nbt.f1D,  # state_zero: np.ndarray,
        nbt.f1D,  # simulated_results: np.ndarray,
        nbt.f,  # q_gas_accu: float,
        nbt.f,  # q_ch4_accu: float,
        nbt.f,  # p_co2_day_mean_accu: float,
        nbt.f,  # p_ch4_day_mean_accu: float,
        nbt.f,  # K_a_co2: float,
        nbt.f,  # K_a_IN: float,
        nbt.f,  # K_w: float,
        nbt.f,  # K_pH_aa: float,
        nbt.f,  # nn_aa: float,
        nbt.f,  # K_pH_h2: float,
        nbt.f,  # n_h2: float,
        nbt.f,  # K_S_IN: float,
        nbt.f,  # K_I_h2_fa: float,
        nbt.f,  # K_H_h2: float,
        nbt.f,  # K_I_h2_c4: float,
        nbt.f,  # K_I_h2_pro: float,
        nbt.f,  # k_m_su: float,
        nbt.f,  # K_S_su: float,
        nbt.f,  # k_m_aa: float,
        nbt.f,  # K_S_aa: float,
        nbt.f,  # k_m_pro: float,
        nbt.f,  # K_S_pro: float,
        nbt.f,  # k_m_fa: float,
        nbt.f,  # K_S_fa: float,
        nbt.f,  # k_m_c4: float,
        nbt.f,  # K_S_c4: float,
        nbt.f,  # k_m_h2: float,
        nbt.f,  # K_S_h2: float,
        nbt.f,  # T_op: float,
        nbt.f,  # p_gas_h2o: float,
        nbt.f,  # V_liq: float,
        nbt.f,  # q_ad: float,
        nbt.f,  # S_h2_in: float,
        nbt.f,  # T_ad: float,
        nbt.f,  # tstep1: float,
        nbt.f,  # tstep0: float,
        # New params
        nbt.f,  # K_a_va: float,
        nbt.f,  # K_a_bu: float,
        nbt.f,  # K_a_pro: float,
        nbt.f,  # K_a_ac: float,
        nbt.f,  # f_h2_su: float,
        nbt.f,  # f_h2_aa: float,
        nbt.f,  # Y_su: float,
        nbt.f,  # Y_aa: float,
        nbt.f,  # Y_fa: float,
        nbt.f,  # Y_c4: float,
        nbt.f,  # Y_pro: float,
        nbt.f,  # k_p: float,
        nbt.f,  # k_L_a: float,
    )
)
def post_treatment(
    state_zero: np.ndarray,
    simulated_results: np.ndarray,
    q_gas_accu: float,
    q_ch4_accu: float,
    p_co2_day_mean_accu: float,
    p_ch4_day_mean_accu: float,
    K_a_co2: float,
    K_a_IN: float,
    K_w: float,
    K_pH_aa: float,
    nn_aa: float,
    K_pH_h2: float,
    n_h2: float,
    K_S_IN: float,
    K_I_h2_fa: float,
    K_H_h2: float,
    K_I_h2_c4: float,
    K_I_h2_pro: float,
    k_m_su: float,
    K_S_su: float,
    k_m_aa: float,
    K_S_aa: float,
    k_m_pro: float,
    K_S_pro: float,
    k_m_fa: float,
    K_S_fa: float,
    k_m_c4: float,
    K_S_c4: float,
    k_m_h2: float,
    K_S_h2: float,
    T_op: float,
    p_gas_h2o: float,
    V_liq: float,
    q_ad: float,
    S_h2_in: float,
    T_ad: float,
    tstep1: float,
    tstep0: float,
    # New params
    K_a_va: float,
    K_a_bu: float,
    K_a_pro: float,
    K_a_ac: float,
    f_h2_su: float,
    f_h2_aa: float,
    Y_su: float,
    Y_aa: float,
    Y_fa: float,
    Y_c4: float,
    Y_pro: float,
    k_p: float,
    k_L_a: float,
):
    # Check if simulations results are all positive
    # They should be but unfortunately negative pressures
    # observed in previous run_adm1 runs for h2 and ch4
    # can only be explained by negative S_h2 and S_ch4

    if np.any(simulated_results < 0.0):
        # If negative result, divide by 2 previous result
        wrong_terms = simulated_results < 0.0
        simulated_results[wrong_terms] = 0.5 * state_zero[wrong_terms]

    (
        S_su,
        S_aa,
        S_fa,
        S_va,
        S_bu,
        S_pro,
        S_ac,
        S_h2,
        S_ch4,
        S_IC,
        S_IN,
        S_I,
        X_c,
        X_ch,
        X_pr,
        X_li,
        X_su,
        X_aa,
        X_fa,
        X_c4,
        X_pro,
        X_ac,
        X_h2,
        X_I,
        S_cation,
        S_anion,
        S_H_ion,
        S_va_ion,
        S_bu_ion,
        S_pro_ion,
        S_ac_ion,
        S_hco3_ion,
        S_nh3,
        S_gas_h2,
        S_gas_ch4,
        S_gas_co2,
    ) = simulated_results

    # Solve DAE states
    eps = 0.0000001
    prevS_H_ion = S_H_ion

    # initial values for Newton-Raphson solver parameter
    shdelta = 1.0
    shgradeq = 1.0
    S_h2delta = 1.0
    S_h2gradeq = 1.0
    tol = 10.0 ** (-12)  # solver accuracy tolerance
    maxIter = 1000  # maximum number of iterations for solver
    i = 1
    j = 1

    ## DAE solver for S_H_ion from Rosen et al. (2006)
    while (shdelta > tol or shdelta < -tol) and (i <= maxIter):
        S_va_ion = K_a_va * S_va / (K_a_va + S_H_ion)
        S_bu_ion = K_a_bu * S_bu / (K_a_bu + S_H_ion)
        S_pro_ion = K_a_pro * S_pro / (K_a_pro + S_H_ion)
        S_ac_ion = K_a_ac * S_ac / (K_a_ac + S_H_ion)
        S_hco3_ion = K_a_co2 * S_IC / (K_a_co2 + S_H_ion)
        S_nh3 = K_a_IN * S_IN / (K_a_IN + S_H_ion)
        shdelta = (
            S_cation
            + (S_IN - S_nh3)
            + S_H_ion
            - S_hco3_ion
            - S_ac_ion / 64.0
            - S_pro_ion / 112.0
            - S_bu_ion / 160.0
            - S_va_ion / 208.0
            - K_w / S_H_ion
            - S_anion
        )
        shgradeq = (
            1
            + K_a_IN * S_IN / ((K_a_IN + S_H_ion) * (K_a_IN + S_H_ion))
            + K_a_co2 * S_IC / ((K_a_co2 + S_H_ion) * (K_a_co2 + S_H_ion))
            + 1.0 / 64.0 * K_a_ac * S_ac / ((K_a_ac + S_H_ion) * (K_a_ac + S_H_ion))
            + 1 / 112.0 * K_a_pro * S_pro / ((K_a_pro + S_H_ion) * (K_a_pro + S_H_ion))
            + 1.0 / 160.0 * K_a_bu * S_bu / ((K_a_bu + S_H_ion) * (K_a_bu + S_H_ion))
            + 1.0 / 208.0 * K_a_va * S_va / ((K_a_va + S_H_ion) * (K_a_va + S_H_ion))
            + K_w / (S_H_ion * S_H_ion)
        )
        S_H_ion = S_H_ion - shdelta / shgradeq
        if S_H_ion <= 0:
            S_H_ion = tol
        i += 1

    ## DAE solver for S_h2 from Rosen et al. (2006)
    while (S_h2delta > tol or S_h2delta < -tol) and (j <= maxIter):
        I_pH_aa = (K_pH_aa**nn_aa) / (prevS_H_ion**nn_aa + K_pH_aa**nn_aa)

        I_pH_h2 = (K_pH_h2**n_h2) / (prevS_H_ion**n_h2 + K_pH_h2**n_h2)
        I_IN_lim = S_IN / (S_IN + K_S_IN)
        I_h2_fa = K_I_h2_fa / (K_I_h2_fa + S_h2)
        I_h2_c4 = K_I_h2_c4 / (K_I_h2_c4 + S_h2)
        I_h2_pro = K_I_h2_pro / (K_I_h2_pro + S_h2)

        I_5 = I_pH_aa * I_IN_lim
        I_6 = I_5
        I_7 = I_pH_aa * I_IN_lim * I_h2_fa
        I_8 = I_pH_aa * I_IN_lim * I_h2_c4
        I_9 = I_8
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro

        I_12 = I_pH_h2 * I_IN_lim
        Rho_5 = k_m_su * (S_su / (K_S_su + S_su)) * X_su * I_5  # Uptake of sugars
        Rho_6 = k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6  # Uptake of amino-acids
        Rho_7 = (
            k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7
        )  # Uptake of LCFA (long-chain fatty acids)
        Rho_8 = (
            k_m_c4
            * (S_va / (K_S_c4 + S_va))
            * X_c4
            * (S_va / (S_bu + S_va + 1e-6))
            * I_8
        )  # Uptake of valerate
        Rho_9 = (
            k_m_c4
            * (S_bu / (K_S_c4 + S_bu))
            * X_c4
            * (S_bu / (S_bu + S_va + 1e-6))
            * I_9
        )  # Uptake of butyrate
        Rho_10 = (
            k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10
        )  # Uptake of propionate
        Rho_12 = k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12  # Uptake of hydrogen
        p_gas_h2 = S_gas_h2 * R * T_ad / 16
        Rho_T_8 = k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2)
        S_h2delta = (
            q_ad / V_liq * (S_h2_in - S_h2)
            + (1.0 - Y_su) * f_h2_su * Rho_5
            + (1.0 - Y_aa) * f_h2_aa * Rho_6
            + (1.0 - Y_fa) * 0.3 * Rho_7
            + (1.0 - Y_c4) * 0.15 * Rho_8
            + (1.0 - Y_c4) * 0.2 * Rho_9
            + (1.0 - Y_pro) * 0.43 * Rho_10
            - Rho_12
            - Rho_T_8
        )
        S_h2gradeq = (
            -1.0 / V_liq * q_ad
            - 3.0
            / 10.0
            * (1.0 - Y_fa)
            * k_m_fa
            * S_fa
            / (K_S_fa + S_fa)
            * X_fa
            * I_pH_aa
            * S_IN
            / (K_S_IN + S_IN)
            * K_I_h2_fa
            / ((K_I_h2_fa + S_h2) ** 2)
            - 3.0
            / 20.0
            * (1.0 - Y_c4)
            * k_m_c4
            * S_va
            * S_va
            / (K_S_c4 + S_va)
            * X_c4
            / (S_bu + S_va + eps)
            * I_pH_aa
            * S_IN
            / (S_IN + K_S_IN)
            * K_I_h2_c4
            / ((K_I_h2_c4 + S_h2) ** 2)
            - 1.0
            / 5.0
            * (1.0 - Y_c4)
            * k_m_c4
            * S_bu
            * S_bu
            / (K_S_c4 + S_bu)
            * X_c4
            / (S_bu + S_va + eps)
            * I_pH_aa
            * S_IN
            / (S_IN + K_S_IN)
            * K_I_h2_c4
            / ((K_I_h2_c4 + S_h2) ** 2)
            - 43.0
            / 100.0
            * (1.0 - Y_pro)
            * k_m_pro
            * S_pro
            / (K_S_pro + S_pro)
            * X_pro
            * I_pH_aa
            * S_IN
            / (S_IN + K_S_IN)
            * K_I_h2_pro
            / ((K_I_h2_pro + S_h2) ** 2)
            - k_m_h2 / (K_S_h2 + S_h2) * X_h2 * I_pH_h2 * S_IN / (S_IN + K_S_IN)
            + k_m_h2
            * S_h2
            / ((K_S_h2 + S_h2) * (K_S_h2 + S_h2))
            * X_h2
            * I_pH_h2
            * S_IN
            / (S_IN + K_S_IN)
            - k_L_a
        )
        S_h2 = S_h2 - S_h2delta / S_h2gradeq
        if S_h2 <= 0:
            S_h2 = tol
        j += 1
    # DAE states solved

    # Algebraic equations
    p_gas_h2 = S_gas_h2 * R * T_op / 16.0
    p_gas_ch4 = S_gas_ch4 * R * T_op / 64.0
    p_gas_co2 = S_gas_co2 * R * T_op
    p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o
    q_gas = k_p * (p_gas - p_atm)
    if q_gas < 0:
        q_gas = 0
    q_gas_accu += q_gas * (tstep1 - tstep0)

    q_ch4 = q_gas * (p_gas_ch4 / p_gas)  # methane flow
    if q_ch4 < 0.0:  ## q_gas is positive, only negative if negative pression...
        q_ch4 = 0.0
    q_ch4_accu += q_ch4 * (tstep1 - tstep0)

    p_co2_day_mean_accu += p_gas_co2
    p_ch4_day_mean_accu += p_gas_ch4

    # state transfer
    state_zero = np.array(
        [
            S_su,
            S_aa,
            S_fa,
            S_va,
            S_bu,
            S_pro,
            S_ac,
            S_h2,
            S_ch4,
            S_IC,
            S_IN,
            S_I,
            X_c,
            X_ch,
            X_pr,
            X_li,
            X_su,
            X_aa,
            X_fa,
            X_c4,
            X_pro,
            X_ac,
            X_h2,
            X_I,
            S_cation,
            S_anion,
            S_H_ion,
            S_va_ion,
            S_bu_ion,
            S_pro_ion,
            S_ac_ion,
            S_hco3_ion,
            S_nh3,
            S_gas_h2,
            S_gas_ch4,
            S_gas_co2,
        ]
    )
    return state_zero, q_gas_accu, q_ch4_accu, p_co2_day_mean_accu, p_ch4_day_mean_accu


def run_adm1(
    influent_state: np.ndarray,
    initial_state: np.ndarray,
    # Digester info
    V_liq:float,
    V_gas:float,
    T_ad :float,
    T_op:float,
    # Parameter values, unpacked
    N_xc: float,
    N_I: float,
    N_aa: float,
    N_bac: float,
    C_xc: float,
    C_sI: float,
    C_ch: float,
    C_pr: float,
    C_li: float,
    C_xI: float,
    C_su: float,
    C_aa: float,
    C_fa: float,
    C_bu: float,
    C_pro: float,
    C_ac: float,
    C_bac: float,
    C_va: float,
    C_ch4: float,
    K_a_va: float,
    K_a_bu: float,
    K_a_pro: float,
    K_a_ac: float,
    f_sI_xc: float,
    f_xI_xc: float,
    f_ch_xc: float,
    f_pr_xc: float,
    f_li_xc: float,
    f_fa_li: float,
    f_h2_su: float,
    f_bu_su: float,
    f_pro_su: float,
    f_ac_su: float,
    f_h2_aa: float,
    f_va_aa: float,
    f_bu_aa: float,
    f_pro_aa: float,
    f_ac_aa: float,
    Y_su: float,
    Y_aa: float,
    Y_fa: float,
    Y_c4: float,
    Y_pro: float,
    Y_ac: float,
    Y_h2: float,
    k_p: float,
    k_L_a: float,
    k_dis: float,
    k_hyd_ch: float,
    k_hyd_pr: float,
    k_hyd_li: float,
    k_m_su: float,
    k_m_aa: float,
    k_m_fa: float,
    k_m_c4: float,
    k_m_pro: float,
    k_m_ac: float,
    k_m_h2: float,
    k_dec: float,
    K_S_IN: float,
    K_S_su: float,
    K_S_aa: float,
    K_S_fa: float,
    K_S_c4: float,
    K_S_pro: float,
    K_S_ac: float,
    K_S_h2: float,
    K_I_h2_fa: float,
    K_I_h2_c4: float,
    K_I_h2_pro: float,
    K_I_nh3: float,
    pH_UL_aa: float,
    pH_LL_aa: float,
    pH_UL_ac: float,
    pH_LL_ac: float,
    pH_UL_h2: float,
    pH_LL_h2: float,
    # PDE solver hyperparameters
    solver_method: str = "LSODA",
    max_step: float = 60.0 / (24.0 * 60.0),
    min_step: float = 10**-6,
    # Further kwargs for PDE solver
    **kwargs,
) -> np.ndarray:
    """
    Runs ADM1 with Python.
    This function code is based on original PyADM1 package (https://github.com/CaptainFerMag/PyADM1)

    ADM1 models an anaerobic digestion process from feed data and a description of
    the microbial populations in the digester (stored in param).

    This implementation is modified to only use numpy arrays and to address issues
    of the original code. It relies on an internal function adm1_ode, which relies on
    regularly updated variables in the function environnement.

    Args:
        param: Description of the microbial population in the digester (calibration parameter)
        influent_state: Multidimensional time series describing what goes inside the digester
        initial_state: Description of what is inside the digester at time 0
        V_liq: Volume of liquid part of the digester in m3
        V_gas: Volume of the gas part of the digester in m3
        T_ad: Temperature in the digester
        T_op: By default, the temperature in the digester.
        solver_method: Type of solver used to integrate between time steps
        log_path: where should run log be written ?
        stop_on_err: Should the simulation continue after a numerical instability as appeared ?

    Returns:
        A multidimensional time series describing the successive digester states. There is one state per day,
        initial state is not stored. The successive digester states are computed by integrating the ADM1
        differential equation system between time t_n and t_{n+1} using the feed information specified at t_{n+1}.
        t_0 is the time information specified by initial_state.

    Remarks:
        Will fail if two feed consecutive feed events are not either in the same day or in consecutive
        days.

    Minor issues:
        Launching scipy.solve_ivp repeatedly seems to be expensive. No simple patch due to DAE
        state solving part.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        warnings.filterwarnings("error", category=UserWarning)

        ###################################################s#############

        delta_1_T = 1.0 / T_base - 1.0 / T_ad
        p_gas_h2o = 0.0313 * exp(5290.0 * delta_1_T)  # bar #0.0557
        K_H_co2 = 0.035 * exp(
            (-19410.0 / (100.0 * R)) * delta_1_T
        )  # Mliq.bar^-1 #0.0271
        K_H_ch4 = 0.0014 * exp(
            (-14240.0 / (100.0 * R)) * delta_1_T
        )  # Mliq.bar^-1 #0.00116
        K_H_h2 = (
            7.8 * 10.0**-4 * exp(-4180 / (100 * R) * delta_1_T)
        )  # Mliq.bar^-1 #7.38*10^-4

        # T_ad depends on time, should be influent_state
        K_w = 10**-14.0 * exp(
            (55900.0 / (100.0 * R)) * delta_1_T
        )  # M #2.08 * 10 ^ -14
        K_H_co2 = 0.035 * exp(
            (-19410.0 / (100.0 * R)) * delta_1_T
        )  # Mliq.bar^-1 #0.0271
        K_H_ch4 = 0.0014 * exp(
            (-14240.0 / (100.0 * R)) * delta_1_T
        )  # Mliq.bar^-1 #0.00116
        K_H_h2 = (
            7.8 * 10**-4 * exp(-4180.0 / (100.0 * R) * delta_1_T)
        )  # Mliq.bar^-1 #7.38*10^-4

        K_pH_aa = 10.0 ** (-1.0 * (pH_LL_aa + pH_UL_aa) / 2.0)
        nn_aa = 3.0 / (
            pH_UL_aa - pH_LL_aa
        )  # we need a differece between N_aa and n_aa to avoid typos and nn_aa refers to the n_aa in BSM2 report
        K_pH_ac = 10.0 ** (-1.0 * (pH_LL_ac + pH_UL_ac) / 2.0)
        n_ac = 3.0 / (pH_UL_ac - pH_LL_ac)
        K_pH_h2 = 10.0 ** (-1.0 * (pH_LL_h2 + pH_UL_h2) / 2.0)
        n_h2 = 3.0 / (pH_UL_h2 - pH_LL_h2)

        K_a_co2 = 10.0**-6.35 * exp(
            (7646.0 / (100.0 * R)) * delta_1_T
        )  # M #4.94 * 10 ^ -7
        K_a_IN = 10.0 ** (-9.25) * exp(
            (51965.0 / (100.0 * R)) * delta_1_T
        )  # M #1.11 * 10 ^ -9

        ## Add equation parameter
        s_1 = (
            -1.0 * C_xc
            + f_sI_xc * C_sI
            + f_ch_xc * C_ch
            + f_pr_xc * C_pr
            + f_li_xc * C_li
            + f_xI_xc * C_xI
        )
        s_2 = -1.0 * C_ch + C_su
        s_3 = -1.0 * C_pr + C_aa
        s_4 = -1.0 * C_li + (1.0 - f_fa_li) * C_su + f_fa_li * C_fa
        s_5 = (
            -1.0 * C_su
            + (1.0 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac)
            + Y_su * C_bac
        )
        s_6 = (
            -1.0 * C_aa
            + (1.0 - Y_aa)
            * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac)
            + Y_aa * C_bac
        )
        s_7 = -1.0 * C_fa + (1.0 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac
        s_8 = (
            -1.0 * C_va
            + (1.0 - Y_c4) * 0.54 * C_pro
            + (1.0 - Y_c4) * 0.31 * C_ac
            + Y_c4 * C_bac
        )
        s_9 = -1.0 * C_bu + (1.0 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac
        s_10 = -1.0 * C_pro + (1.0 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac
        s_11 = -1.0 * C_ac + (1.0 - Y_ac) * C_ch4 + Y_ac * C_bac
        s_12 = (1.0 - Y_h2) * C_ch4 + Y_h2 * C_bac
        s_13 = -1.0 * C_bac + C_xc

        ## END OF PARAM UNPACKING

        ############################################

        ## Create initial values
        t0 = initial_state[0]
        # Initial state elements follow the order of pred_col in helper_pd_np file
        # For internal use, the pH information is converted to concentration
        # Part of the information in initial state is disregarded (S_co2, S_nh4_ion,
        # q_gas, q_ch4, p_ch4, p_co2, VS & VSR)

        state_zero = initial_state[1:37]
        state_zero[26] = 10 ** (-state_zero[26])  # Converting pH to concentration

        t = influent_state[:, 0]

        # Number of days in feed
        day_begin = int(t0) + 1
        day_end = int(t[len(t) - 1])

        n_days = day_end - day_begin + 1

        ## Create empty accu for results
        simulate_results = np.full(shape=(n_days, 45), fill_value=np.nan)
        simulate_results[0, 1:37] = state_zero
        # columns 36, 37 and 42 are created in the end.
        # For columns 38 to 41, need to compute intermediary values
        S_gas_h2 = state_zero[33]
        S_gas_ch4 = state_zero[34]
        S_gas_co2 = state_zero[35]
        p_gas_h2 = S_gas_h2 * R * T_op / 16
        p_gas_ch4 = S_gas_ch4 * R * T_op / 64
        p_gas_co2 = S_gas_co2 * R * T_op
        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o

        simulate_results[0, 39:43] = [
            k_p * (p_gas - p_atm),  # q_gas
            k_p * (p_gas - p_atm) * (p_gas_ch4 / p_gas),  # q_ch4
            p_gas_ch4,  # p_ch4
            p_gas_co2,  # p_co2
        ]

        ########################################################################
        ## ADM1 Ordinary Differential Equation

        ## Simulate
        # If/else to avoid UserWarning trouble from min_step for non LSODA solver method

        if solver_method == "LSODA":

            def simulate(
                tstep,
                state_zero,
            ):
                loc_adm1_ode = partial(
                    adm1_ode,
                    K_pH_aa=K_pH_aa,
                    nn_aa=nn_aa,
                    K_pH_h2=K_pH_h2,
                    n_h2=n_h2,
                    K_pH_ac=K_pH_ac,
                    n_ac=n_ac,
                    K_S_IN=K_S_IN,
                    K_I_h2_fa=K_I_h2_fa,
                    K_H_h2=K_H_h2,
                    K_I_h2_c4=K_I_h2_c4,
                    K_I_h2_pro=K_I_h2_pro,
                    K_I_nh3=K_I_nh3,
                    k_dis=k_dis,
                    k_hyd_ch=k_hyd_ch,
                    k_hyd_pr=k_hyd_pr,
                    k_hyd_li=k_hyd_li,
                    k_m_su=k_m_su,
                    K_S_su=K_S_su,
                    k_m_aa=k_m_aa,
                    K_S_aa=K_S_aa,
                    k_m_pro=k_m_pro,
                    K_S_pro=K_S_pro,
                    k_m_fa=k_m_fa,
                    K_S_fa=K_S_fa,
                    k_m_c4=k_m_c4,
                    K_S_c4=K_S_c4,
                    k_m_ac=k_m_ac,
                    K_S_ac=K_S_ac,
                    k_dec=k_dec,
                    k_m_h2=k_m_h2,
                    K_S_h2=K_S_h2,
                    T_op=T_op,
                    p_gas_h2o=p_gas_h2o,
                    V_liq=V_liq,
                    q_ad=q_ad,
                    K_H_ch4=K_H_ch4,
                    K_H_co2=K_H_co2,
                    S_su_in=S_su_in,
                    S_aa_in=S_aa_in,
                    S_fa_in=S_fa_in,
                    S_va_in=S_va_in,
                    S_bu_in=S_bu_in,
                    S_pro_in=S_pro_in,
                    S_ac_in=S_ac_in,
                    S_ch4_in=S_ch4_in,
                    s_1=s_1,
                    s_2=s_2,
                    s_3=s_3,
                    s_4=s_4,
                    s_5=s_5,
                    s_6=s_6,
                    s_7=s_7,
                    s_8=s_8,
                    s_9=s_9,
                    s_10=s_10,
                    s_11=s_11,
                    s_12=s_12,
                    s_13=s_13,
                    V_gas=V_gas,
                    S_IC_in=S_IC_in,
                    S_IN_in=S_IN_in,
                    S_I_in=S_I_in,
                    X_c_in=X_c_in,
                    X_ch_in=X_ch_in,
                    X_pr_in=X_pr_in,
                    X_li_in=X_li_in,
                    X_su_in=X_su_in,
                    X_aa_in=X_aa_in,
                    X_fa_in=X_fa_in,
                    X_c4_in=X_c4_in,
                    X_pro_in=X_pro_in,
                    X_ac_in=X_ac_in,
                    X_h2_in=X_h2_in,
                    X_I_in=X_I_in,
                    S_cation_in=S_cation_in,
                    S_anion_in=S_anion_in,
                    N_xc=N_xc,
                    N_I=N_I,
                    N_aa=N_aa,
                    N_bac=N_bac,
                    f_sI_xc=f_sI_xc,
                    f_xI_xc=f_xI_xc,
                    f_ch_xc=f_ch_xc,
                    f_pr_xc=f_pr_xc,
                    f_li_xc=f_li_xc,
                    f_fa_li=f_fa_li,
                    f_bu_su=f_bu_su,
                    f_pro_su=f_pro_su,
                    f_ac_su=f_ac_su,
                    f_va_aa=f_va_aa,
                    f_bu_aa=f_bu_aa,
                    f_pro_aa=f_pro_aa,
                    f_ac_aa=f_ac_aa,
                    Y_su=Y_su,
                    Y_aa=Y_aa,
                    Y_fa=Y_fa,
                    Y_c4=Y_c4,
                    Y_pro=Y_pro,
                    Y_ac=Y_ac,
                    Y_h2=Y_h2,
                    k_p=k_p,
                    k_L_a=k_L_a,
                )
                r = scipy.integrate.solve_ivp(
                    loc_adm1_ode,
                    tstep,
                    state_zero,
                    method="LSODA",
                    max_step=max_step,
                    min_step=min_step,
                    **kwargs,
                )
                return r.y[:, -1]

        else:

            def simulate(
                tstep,
                state_zero,
            ):
                loc_adm1_ode = partial(
                    adm1_ode,
                    K_pH_aa=K_pH_aa,
                    nn_aa=nn_aa,
                    K_pH_h2=K_pH_h2,
                    n_h2=n_h2,
                    K_pH_ac=K_pH_ac,
                    n_ac=n_ac,
                    K_S_IN=K_S_IN,
                    K_I_h2_fa=K_I_h2_fa,
                    K_H_h2=K_H_h2,
                    K_I_h2_c4=K_I_h2_c4,
                    K_I_h2_pro=K_I_h2_pro,
                    K_I_nh3=K_I_nh3,
                    k_dis=k_dis,
                    k_hyd_ch=k_hyd_ch,
                    k_hyd_pr=k_hyd_pr,
                    k_hyd_li=k_hyd_li,
                    k_m_su=k_m_su,
                    K_S_su=K_S_su,
                    k_m_aa=k_m_aa,
                    K_S_aa=K_S_aa,
                    k_m_pro=k_m_pro,
                    K_S_pro=K_S_pro,
                    k_m_fa=k_m_fa,
                    K_S_fa=K_S_fa,
                    k_m_c4=k_m_c4,
                    K_S_c4=K_S_c4,
                    k_m_ac=k_m_ac,
                    K_S_ac=K_S_ac,
                    k_dec=k_dec,
                    k_m_h2=k_m_h2,
                    K_S_h2=K_S_h2,
                    T_op=T_op,
                    p_gas_h2o=p_gas_h2o,
                    V_liq=V_liq,
                    q_ad=q_ad,
                    K_H_ch4=K_H_ch4,
                    K_H_co2=K_H_co2,
                    S_su_in=S_su_in,
                    S_aa_in=S_aa_in,
                    S_fa_in=S_fa_in,
                    S_va_in=S_va_in,
                    S_bu_in=S_bu_in,
                    S_pro_in=S_pro_in,
                    S_ac_in=S_ac_in,
                    S_ch4_in=S_ch4_in,
                    s_1=s_1,
                    s_2=s_2,
                    s_3=s_3,
                    s_4=s_4,
                    s_5=s_5,
                    s_6=s_6,
                    s_7=s_7,
                    s_8=s_8,
                    s_9=s_9,
                    s_10=s_10,
                    s_11=s_11,
                    s_12=s_12,
                    s_13=s_13,
                    V_gas=V_gas,
                    S_IC_in=S_IC_in,
                    S_IN_in=S_IN_in,
                    S_I_in=S_I_in,
                    X_c_in=X_c_in,
                    X_ch_in=X_ch_in,
                    X_pr_in=X_pr_in,
                    X_li_in=X_li_in,
                    X_su_in=X_su_in,
                    X_aa_in=X_aa_in,
                    X_fa_in=X_fa_in,
                    X_c4_in=X_c4_in,
                    X_pro_in=X_pro_in,
                    X_ac_in=X_ac_in,
                    X_h2_in=X_h2_in,
                    X_I_in=X_I_in,
                    S_cation_in=S_cation_in,
                    S_anion_in=S_anion_in,
                    N_xc=N_xc,
                    N_I=N_I,
                    N_aa=N_aa,
                    N_bac=N_bac,
                    f_sI_xc=f_sI_xc,
                    f_xI_xc=f_xI_xc,
                    f_ch_xc=f_ch_xc,
                    f_pr_xc=f_pr_xc,
                    f_li_xc=f_li_xc,
                    f_fa_li=f_fa_li,
                    f_bu_su=f_bu_su,
                    f_pro_su=f_pro_su,
                    f_ac_su=f_ac_su,
                    f_va_aa=f_va_aa,
                    f_bu_aa=f_bu_aa,
                    f_pro_aa=f_pro_aa,
                    f_ac_aa=f_ac_aa,
                    Y_su=Y_su,
                    Y_aa=Y_aa,
                    Y_fa=Y_fa,
                    Y_c4=Y_c4,
                    Y_pro=Y_pro,
                    Y_ac=Y_ac,
                    Y_h2=Y_h2,
                    k_p=k_p,
                    k_L_a=k_L_a,
                )
                r = scipy.integrate.solve_ivp(
                    loc_adm1_ode,
                    tstep,
                    state_zero,
                    method=solver_method,
                    max_step=max_step,
                    **kwargs,
                )
                return r.y[:, -1]

        ################################################################
        ## Main loop
        n = 0
        loc_day = int(t0) + 1  # Keep track of when to save information
        count_day = 0
        q_ch4_accu = 0
        q_gas_accu = 0
        p_co2_day_mean_accu = 0
        p_ch4_day_mean_accu = 0
        count_in_day = 0

        for u in t:
            count_in_day += 1
            ## Set up influent state
            (
                S_su_in,
                S_aa_in,
                S_fa_in,
                S_va_in,
                S_bu_in,
                S_pro_in,
                S_ac_in,
                S_h2_in,
                S_ch4_in,
                S_IC_in,
                S_IN_in,
                S_I_in,
                X_c_in,
                X_ch_in,
                X_pr_in,
                X_li_in,
                X_su_in,
                X_aa_in,
                X_fa_in,
                X_c4_in,
                X_pro_in,
                X_ac_in,
                X_h2_in,
                X_I_in,
                S_cation_in,
                S_anion_in,
                q_ad,
            ) = influent_state[
                n, 1:28
            ]  # 0 is for time and not used here

            # Span for next time step
            tstep = [t0, u]

            # Run integration to next step
            # Force first compilation

            simulated_results = simulate(tstep=tstep, state_zero=state_zero)

            # Check if simulations results are all positive
            # They should be but unfortunately negative pressures
            # observed in previous run_adm1 runs for h2 and ch4
            # can only be explained by negative S_h2 and S_ch4

            (
                state_zero,
                q_gas_accu,
                q_ch4_accu,
                p_co2_day_mean_accu,
                p_ch4_day_mean_accu,
            ) = post_treatment(
                state_zero=state_zero,
                simulated_results=simulated_results,
                q_gas_accu=q_gas_accu,
                q_ch4_accu=q_ch4_accu,
                p_co2_day_mean_accu=p_co2_day_mean_accu,
                p_ch4_day_mean_accu=p_ch4_day_mean_accu,
                K_a_co2=K_a_co2,
                K_a_IN=K_a_IN,
                K_w=K_w,
                K_pH_aa=K_pH_aa,
                nn_aa=nn_aa,
                K_pH_h2=K_pH_h2,
                n_h2=n_h2,
                K_S_IN=K_S_IN,
                K_I_h2_fa=K_I_h2_fa,
                K_H_h2=K_H_h2,
                K_I_h2_c4=K_I_h2_c4,
                K_I_h2_pro=K_I_h2_pro,
                k_m_su=k_m_su,
                K_S_su=K_S_su,
                k_m_aa=k_m_aa,
                K_S_aa=K_S_aa,
                k_m_pro=k_m_pro,
                K_S_pro=K_S_pro,
                k_m_fa=k_m_fa,
                K_S_fa=K_S_fa,
                k_m_c4=k_m_c4,
                K_S_c4=K_S_c4,
                k_m_h2=k_m_h2,
                K_S_h2=K_S_h2,
                T_op=T_op,
                p_gas_h2o=p_gas_h2o,
                V_liq=V_liq,
                q_ad=q_ad,
                S_h2_in=S_h2_in,
                T_ad=T_ad,
                tstep1=tstep[1],
                tstep0=tstep[0],
                K_a_va=K_a_va,
                K_a_bu=K_a_bu,
                K_a_pro=K_a_pro,
                K_a_ac=K_a_ac,
                f_h2_su=f_h2_su,
                f_h2_aa=f_h2_aa,
                Y_su=Y_su,
                Y_aa=Y_aa,
                Y_fa=Y_fa,
                Y_c4=Y_c4,
                Y_pro=Y_pro,
                k_p=k_p,
                k_L_a=k_L_a,
            )

            if int(u) >= loc_day:
                simulate_results[count_day, 1:37] = state_zero
                simulate_results[count_day, 39:43] = [
                    q_gas_accu,
                    q_ch4_accu,
                    p_ch4_day_mean_accu / count_in_day,
                    p_co2_day_mean_accu / count_in_day,
                ]
                count_day = count_day + 1
                loc_day = int(u) + 1  # Should be loc_day + 1 as well

                ## Refresh accus
                q_gas_accu = 0
                q_ch4_accu = 0
                p_ch4_day_mean_accu = 0
                p_co2_day_mean_accu = 0
                count_in_day = 0

            n = n + 1
            t0 = u
        ## END OF LOOP
        simulate_results[:, 27] = -np.log10(
            simulate_results[:, 27]
        )  # Transforming back pH from concentration
        simulate_results[:, 37] = (
            simulate_results[:, 10] - simulate_results[:, 32]  # S_co2 creation
        )
        simulate_results[:, 38] = (
            simulate_results[:, 11] - simulate_results[:, 33]
        )  # S_nh4_ion creation

        # VSR computation
        VS_dig = dig_states_vs(simulate_results)  # type: ignore
        VS_in = feed_vs(influent_state, per_day=True, day_begin=day_begin)
        simulate_results[:, 43] = VS_dig
        simulate_results[:, 44] = (VS_in - VS_dig) / VS_in

        # Add time to results
        time_sim = np.arange(day_begin, day_end + 1, 1)
        simulate_results[:, 0] = time_sim

        return simulate_results
