import pandas as pd


class ADM1Param:
    """ADM1 parameter class

    Parameter information is stored in a dictionnary which can be accessed
    through the "param" property. Checks are performed to force stoichiometry
    and order constraints on the parameters.

    Compared to v1 of pyadm1 package, a lot more parameters are included in
    ADM1Param and are no longer fixed to their default values, to accomodate
    more situations.

    Default values are inferred from Rosen and Jeppsson (doi: 10.2166/wst.2006.521)
    """

    default_param = {
        "N_xc": 0.002,
        "N_I": 0.002,
        "N_aa": 0.007,
        "N_bac": 0.006,
        "C_xc": 0.03,
        "C_sI": 0.03,
        "C_ch": 0.03,
        "C_pr": 0.03,
        "C_li": 0.023,
        "C_xI": 0.03,
        "C_su": 0.03,
        "C_aa": 0.03,
        "C_fa": 0.0217,
        "C_bu": 0.025,
        "C_pro": 0.0268,
        "C_ac": 0.0313,
        "C_bac": 0.0313,
        "C_va": 0.024,
        "C_ch4": 0.0156,
        "K_a_va": 0.0000138038426,
        "K_a_bu": 0.00001513561248,
        "K_a_pro": 0.000013182567,
        "K_a_ac": 0.000017378008287,
        "f_sI_xc": 0.1,
        "f_xI_xc": 0.2,
        "f_ch_xc": 0.2,
        "f_pr_xc": 0.2,
        "f_li_xc": 0.3,
        "f_fa_li": 0.95,
        "f_h2_su": 0.19,
        "f_bu_su": 0.13,
        "f_pro_su": 0.27,
        "f_ac_su": 0.41,
        "f_h2_aa": 0.06,
        "f_va_aa": 0.23,
        "f_bu_aa": 0.26,
        "f_pro_aa": 0.05,
        "f_ac_aa": 0.40,
        "Y_su": 0.1,  # kg CODX kg CODS-1
        "Y_aa": 0.08,  # kg CODX kg CODS-1
        "Y_fa": 0.06,  # kg CODX kg CODS-1
        "Y_c4": 0.06,  # kg CODX kg CODS-1
        "Y_pro": 0.04,  # kg CODX kg CODS-1
        "Y_ac": 0.05,  # kg CODX kg CODS-1
        "Y_h2": 0.06,  # kg CODX kg CODS-1
        # Other constants
        "k_p": 50000,  # M3 Day-1 bar-1
        "k_L_a": 200,  # Day-1
        "k_dis": 0.5,
        "k_hyd_ch": 10,
        "k_hyd_pr": 10,
        "k_hyd_li": 10,
        "k_m_su": 30,
        "k_m_aa": 50,
        "k_m_fa": 6,
        "k_m_c4": 20,
        "k_m_pro": 13,
        "k_m_ac": 8,
        "k_m_h2": 35,
        "k_dec": 0.02,
        "K_S_IN": 0.0001,
        "K_S_su": 0.5,
        "K_S_aa": 0.3,
        "K_S_fa": 0.4,
        "K_S_c4": 0.2,
        "K_S_pro": 0.1,
        "K_S_ac": 0.15,
        "K_S_h2": 7e-6,
        "K_I_h2_fa": 5e-6,
        "K_I_h2_c4": 1e-5,
        "K_I_h2_pro": 3.5e-6,
        "K_I_nh3": 0.0018,
        "pH_UL_aa": 5.5,
        "pH_LL_aa": 4.0,
        "pH_UL_ac": 6.0,
        "pH_LL_ac": 5.0,
        "pH_UL_h2": 6.0,
        "pH_LL_h2": 5.0,
    }
    par_names = default_param.keys()
    tol_stoichio = 10**-8

    def __init__(self, d: dict[str, float]):
        self.param = d

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, d: dict[str, float]):
        param = self.default_param.copy()
        extra = set(d.keys()).difference(self.par_names)
        if extra:
            raise ValueError(f"Could not interpret keys {extra}")

        param.update(d)
        # Perform checks
        check_param(param, tol_stoichio=self.tol_stoichio)
        self._param = param

    @property
    def N_xc(self):
        return self._param["N_xc"]

    @N_xc.setter
    def N_xc(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["N_xc"] = value

    @property
    def N_I(self):
        return self._param["N_I"]

    @N_I.setter
    def N_I(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["N_I"] = value

    @property
    def N_aa(self):
        return self._param["N_aa"]

    @N_aa.setter
    def N_aa(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["N_aa"] = value

    @property
    def N_bac(self):
        return self._param["N_bac"]

    @N_bac.setter
    def N_bac(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["N_bac"] = value

    @property
    def C_xc(self):
        return self._param["C_xc"]

    @C_xc.setter
    def C_xc(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_xc"] = value

    @property
    def C_sI(self):
        return self._param["C_sI"]

    @C_sI.setter
    def C_sI(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_sI"] = value

    @property
    def C_ch(self):
        return self._param["C_ch"]

    @C_ch.setter
    def C_ch(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_ch"] = value

    @property
    def C_pr(self):
        return self._param["C_pr"]

    @C_pr.setter
    def C_pr(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_pr"] = value

    @property
    def C_li(self):
        return self._param["C_li"]

    @C_li.setter
    def C_li(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_li"] = value

    @property
    def C_xI(self):
        return self._param["C_xI"]

    @C_xI.setter
    def C_xI(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_xI"] = value

    @property
    def C_su(self):
        return self._param["C_su"]

    @C_su.setter
    def C_su(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_su"] = value

    @property
    def C_aa(self):
        return self._param["C_aa"]

    @C_aa.setter
    def C_aa(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_aa"] = value

    @property
    def C_fa(self):
        return self._param["C_fa"]

    @C_fa.setter
    def C_fa(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_fa"] = value

    @property
    def C_bu(self):
        return self._param["C_bu"]

    @C_bu.setter
    def C_bu(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_bu"] = value

    @property
    def C_pro(self):
        return self._param["C_pro"]

    @C_pro.setter
    def C_pro(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_pro"] = value

    @property
    def C_ac(self):
        return self._param["C_ac"]

    @C_ac.setter
    def C_ac(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_ac"] = value

    @property
    def C_bac(self):
        return self._param["C_bac"]

    @C_bac.setter
    def C_bac(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_bac"] = value

    @property
    def C_va(self):
        return self._param["C_va"]

    @C_va.setter
    def C_va(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_va"] = value

    @property
    def C_ch4(self):
        return self._param["C_ch4"]

    @C_ch4.setter
    def C_ch4(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["C_ch4"] = value

    @property
    def K_a_va(self):
        return self._param["K_a_va"]

    @K_a_va.setter
    def K_a_va(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_a_va"] = value

    @property
    def K_a_bu(self):
        return self._param["K_a_bu"]

    @K_a_bu.setter
    def K_a_bu(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_a_bu"] = value

    @property
    def K_a_pro(self):
        return self._param["K_a_pro"]

    @K_a_pro.setter
    def K_a_pro(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_a_pro"] = value

    @property
    def K_a_ac(self):
        return self._param["K_a_ac"]

    @K_a_ac.setter
    def K_a_ac(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_a_ac"] = value

    @property
    def Y_su(self):
        return self._param["Y_su"]

    @Y_su.setter
    def Y_su(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["Y_su"] = value

    @property
    def Y_aa(self):
        return self._param["Y_aa"]

    @Y_aa.setter
    def Y_aa(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["Y_aa"] = value

    @property
    def Y_fa(self):
        return self._param["Y_fa"]

    @Y_fa.setter
    def Y_fa(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["Y_fa"] = value

    @property
    def Y_c4(self):
        return self._param["Y_c4"]

    @Y_c4.setter
    def Y_c4(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["Y_c4"] = value

    @property
    def Y_pro(self):
        return self._param["Y_pro"]

    @Y_pro.setter
    def Y_pro(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["Y_pro"] = value

    @property
    def Y_ac(self):
        return self._param["Y_ac"]

    @Y_ac.setter
    def Y_ac(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["Y_ac"] = value

    @property
    def Y_h2(self):
        return self._param["Y_h2"]

    @Y_h2.setter
    def Y_h2(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["Y_h2"] = value

    @property
    def k_p(self):
        return self._param["k_p"]

    @k_p.setter
    def k_p(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_p"] = value

    @property
    def k_L_a(self):
        return self._param["k_L_a"]

    @k_L_a.setter
    def k_L_a(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_L_a"] = value

    @property
    def k_dis(self):
        return self._param["k_dis"]

    @k_dis.setter
    def k_dis(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_dis"] = value

    @property
    def k_hyd_ch(self):
        return self._param["k_hyd_ch"]

    @k_hyd_ch.setter
    def k_hyd_ch(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_hyd_ch"] = value

    @property
    def k_hyd_pr(self):
        return self._param["k_hyd_pr"]

    @k_hyd_pr.setter
    def k_hyd_pr(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_hyd_pr"] = value

    @property
    def k_hyd_li(self):
        return self._param["k_hyd_li"]

    @k_hyd_li.setter
    def k_hyd_li(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_hyd_li"] = value

    @property
    def k_m_su(self):
        return self._param["k_m_su"]

    @k_m_su.setter
    def k_m_su(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_m_su"] = value

    @property
    def k_m_aa(self):
        return self._param["k_m_aa"]

    @k_m_aa.setter
    def k_m_aa(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_m_aa"] = value

    @property
    def k_m_fa(self):
        return self._param["k_m_fa"]

    @k_m_fa.setter
    def k_m_fa(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_m_fa"] = value

    @property
    def k_m_c4(self):
        return self._param["k_m_c4"]

    @k_m_c4.setter
    def k_m_c4(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_m_c4"] = value

    @property
    def k_m_pro(self):
        return self._param["k_m_pro"]

    @k_m_pro.setter
    def k_m_pro(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_m_pro"] = value

    @property
    def k_m_ac(self):
        return self._param["k_m_ac"]

    @k_m_ac.setter
    def k_m_ac(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_m_ac"] = value

    @property
    def k_m_h2(self):
        return self._param["k_m_h2"]

    @k_m_h2.setter
    def k_m_h2(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_m_h2"] = value

    @property
    def k_dec(self):
        return self._param["k_dec"]

    @k_dec.setter
    def k_dec(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["k_dec"] = value

    @property
    def K_S_IN(self):
        return self._param["K_S_IN"]

    @K_S_IN.setter
    def K_S_IN(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_S_IN"] = value

    @property
    def K_S_su(self):
        return self._param["K_S_su"]

    @K_S_su.setter
    def K_S_su(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_S_su"] = value

    @property
    def K_S_aa(self):
        return self._param["K_S_aa"]

    @K_S_aa.setter
    def K_S_aa(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_S_aa"] = value

    @property
    def K_S_fa(self):
        return self._param["K_S_fa"]

    @K_S_fa.setter
    def K_S_fa(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_S_fa"] = value

    @property
    def K_S_c4(self):
        return self._param["K_S_c4"]

    @K_S_c4.setter
    def K_S_c4(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_S_c4"] = value

    @property
    def K_S_pro(self):
        return self._param["K_S_pro"]

    @K_S_pro.setter
    def K_S_pro(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_S_pro"] = value

    @property
    def K_S_ac(self):
        return self._param["K_S_ac"]

    @K_S_ac.setter
    def K_S_ac(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_S_ac"] = value

    @property
    def K_S_h2(self):
        return self._param["K_S_h2"]

    @K_S_h2.setter
    def K_S_h2(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_S_h2"] = value

    @property
    def K_I_h2_fa(self):
        return self._param["K_I_h2_fa"]

    @K_I_h2_fa.setter
    def K_I_h2_fa(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_I_h2_fa"] = value

    @property
    def K_I_h2_c4(self):
        return self._param["K_I_h2_c4"]

    @K_I_h2_c4.setter
    def K_I_h2_c4(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_I_h2_c4"] = value

    @property
    def K_I_h2_pro(self):
        return self._param["K_I_h2_pro"]

    @K_I_h2_pro.setter
    def K_I_h2_pro(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_I_h2_pro"] = value

    @property
    def K_I_nh3(self):
        return self._param["K_I_nh3"]

    @K_I_nh3.setter
    def K_I_nh3(self, value: float):
        assert isinstance(value, float)
        assert value >= 0.0
        self._param["K_I_nh3"] = value

    @property
    def f_sI_xc(self):
        return self._param["f_sI_xc"]

    @property
    def f_xI_xc(self):
        return self._param["f_xI_xc"]

    @property
    def f_ch_xc(self):
        return self._param["f_ch_xc"]

    @property
    def f_pr_xc(self):
        return self._param["f_pr_xc"]

    @property
    def f_li_xc(self):
        return self._param["f_li_xc"]

    @property
    def f_fa_li(self):
        return self._param["f_fa_li"]

    @f_fa_li.setter
    def f_fa_li(self, value):
        assert isinstance(value, float)
        assert value >= 0.0 & value <= 1.0
        self._param["K_I_nh3"] = value

    @property
    def f_h2_su(self):
        return self._param["f_h2_su"]

    @property
    def f_bu_su(self):
        return self._param["f_bu_su"]

    @property
    def f_pro_su(self):
        return self._param["f_pro_su"]

    @property
    def f_ac_su(self):
        return self._param["f_ac_su"]

    @property
    def f_h2_aa(self):
        return self._param["f_h2_aa"]

    @property
    def f_va_aa(self):
        return self._param["f_va_aa"]

    @property
    def f_bu_aa(self):
        return self._param["f_bu_aa"]

    @property
    def f_pro_aa(self):
        return self._param["f_pro_aa"]

    @property
    def f_ac_aa(self):
        return self._param["f_ac_aa"]

    def updt_f_aa(
        self,
        f_h2_aa: float,
        f_va_aa: float,
        f_bu_aa: float,
        f_pro_aa: float,
        f_ac_aa: float,
    ):
        assert (
            abs(f_h2_aa + f_va_aa + f_bu_aa + f_pro_aa + f_ac_aa - 1.0)
            <= self.tol_stoichio
        ), "Stoichiometry parameters passed do not sum to 1.0"
        self._param.update(
            {
                "f_h2_aa": f_h2_aa,
                "f_va_a": f_va_aa,
                "f_bu_aa": f_bu_aa,
                "f_pro_aa": f_pro_aa,
                "f_ac_aa": f_ac_aa,
            }
        )

    def updt_f_xc(
        self,
        f_sI_xc: float,
        f_xI_xc: float,
        f_ch_xc: float,
        f_pr_xc: float,
        f_li_xc: float,
    ):
        assert (
            abs(f_sI_xc + f_xI_xc + f_ch_xc + f_pr_xc + f_li_xc - 1.0)
            <= self.tol_stoichio
        ), "Stoichiometry parameters passed do not sum to 1.0"
        self._param.update(
            {
                "f_sI_xc": f_sI_xc,
                "f_xI_xc": f_xI_xc,
                "f_ch_xc": f_ch_xc,
                "f_pr_xc": f_pr_xc,
                "f_li_xc": f_li_xc,
            }
        )

    def updt_f_su(
        self, f_h2_su: float, f_bu_su: float, f_pro_su: float, f_ac_su: float
    ):
        assert (
            abs(f_h2_su + f_bu_su + f_pro_su + f_ac_su - 1.0) <= self.tol_stoichio
        ), "Stoichiometry parameters passed do not sum to 1.0"
        self._param.update(
            {
                "f_h2_su": f_h2_su,
                "f_bu_su": f_bu_su,
                "f_pro_su": f_pro_su,
                "f_ac_su": f_ac_su,
            }
        )

    @property
    def pH_UL_aa(self):
        return self._param["pH_UL_aa"]

    @pH_UL_aa.setter
    def pH_UL_aa(self, value: float):
        assert isinstance(value, float)
        assert value >= self.pH_LL_aa
        self._param["pH_UL_aa"] = value

    @property
    def pH_LL_aa(self):
        return self._param["pH_LL_aa"]

    @pH_LL_aa.setter
    def pH_LL_aa(self, value: float):
        assert isinstance(value, float)
        assert (value >= 0.0) & (value <= self.pH_UL_aa)
        self._param["pH_LL_aa"] = value

    @property
    def pH_UL_ac(self):
        return self._param["pH_UL_ac"]

    @pH_UL_ac.setter
    def pH_UL_ac(self, value: float):
        assert isinstance(value, float)
        assert value >= self.pH_LL_ac
        self._param["pH_UL_ac"] = value

    @property
    def pH_LL_ac(self):
        return self._param["pH_LL_ac"]

    @pH_LL_ac.setter
    def pH_LL_ac(self, value: float):
        assert isinstance(value, float)
        assert (value >= 0.0) & (value <= self.pH_UL_ac)
        self._param["pH_LL_ac"] = value

    @property
    def pH_UL_h2(self):
        return self._param["pH_UL_h2"]

    @pH_UL_h2.setter
    def pH_UL_h2(self, value: float):
        assert isinstance(value, float)
        assert value >= self.pH_LL_h2
        self._param["pH_UL_h2"] = value

    @property
    def pH_LL_h2(self):
        return self._param["pH_LL_h2"]

    @pH_LL_h2.setter
    def pH_LL_h2(self, value: float):
        assert isinstance(value, float)
        assert (value >= 0.0) & (value <= self.pH_UL_h2)
        self._param["pH_LL_h2"] = value

    def __repr__(self):
        return self._param.__repr__()

    def save(self, path):
        pd.Series(self._param).to_json(path, orient="index")


def check_param(param: dict, tol_stoichio: float = 10**-8) -> None:

    err_msg = []

    # Check positivity
    if any([x < 0.0 for x in param.values()]):
        err_msg.append("Found negative parameter")

    # pH check
    if param["pH_LL_aa"] >= param["pH_UL_aa"]:
        err_msg.append("pH inhibition for aa error: upper limit <= lower limit")
    if param["pH_LL_ac"] >= param["pH_UL_ac"]:
        err_msg.append("pH inhibition for ac error: upper limit <= lower limit")
    if param["pH_LL_h2"] >= param["pH_UL_h2"]:
        err_msg.append("pH inhibition for h2 error: upper limit <= lower limit")

    if (
        abs(
            param["f_sI_xc"]
            + param["f_xI_xc"]
            + param["f_ch_xc"]
            + param["f_pr_xc"]
            + param["f_li_xc"]
            - 1
        )
        > tol_stoichio
    ):
        err_msg.append("Stoichiometry for xc does not sum to 1.0")

    if (
        abs(
            param["f_h2_su"]
            + param["f_bu_su"]
            + param["f_pro_su"]
            + param["f_ac_su"]
            - 1.0
        )
        > tol_stoichio
    ):
        err_msg.append("Stoichiometry for su does not sum to 1.0")

    if (
        abs(
            param["f_h2_aa"]
            + param["f_va_aa"]
            + param["f_bu_aa"]
            + param["f_pro_aa"]
            + param["f_ac_aa"]
            - 1.0
        )
        > tol_stoichio
    ):
        err_msg.append("Stoichiometry for aa does not sum to 1.0")

    if err_msg:
        err_str = "Invalid parameter:\n" + "\n".join(err_msg)
        raise ValueError(err_str)


def load_dig_param(path: str) -> ADM1Param:
    """
    Loads a digester parameter from a JSON file.
    """
    param_pd = pd.read_json(path, orient="index", typ="Series")
    return ADM1Param(param_pd.to_dict())  # pylint: disable=E1101
