"""
Class for Digester Information.

The necessary information about the digester configuration for the ADM1 routine to work is the
liquid phase volume, gas phase volume and Temperature.

The digester information can be loaded from a json file using load_dig_info.
The digester information can be saved to a json file using the .save method.
"""

import json
from typing import Optional


class DigesterInformation:
    """
    Class for Digester Information.

    Attributes:
        V_liq, the volume of the liquid phase in M3
        V_gas, the volume of the gas phase in M3
        T_ad, the volume inside the digester in Kelvin
    """

    def __init__(
        self, V_liq: float, V_gas: float, T_ad: float, T_op: Optional[float] = None
    ):
        assert V_liq > 0, "The liquid phase volume must be strictly positive"
        assert V_gas > 0, "The gas phase volume must be strictly positive"
        assert (
            T_ad > 0
        ), "The temperature of the digester must be strictly positive (in Kelvin)"

        self.V_liq = float(V_liq)
        self.V_gas = float(V_gas)
        self.T_ad = float(T_ad)
        if T_op is None:
            T_op = T_ad
        self.T_op = float(T_op)

    @property
    def V_liq(self):
        return self._V_liq

    @V_liq.setter
    def V_liq(self, val):
        if val <= 0.0:
            raise ValueError(f"V_liq should be positive (passed {val})")
        self._V_liq = float(val)

    @property
    def V_gas(self):
        return self._V_gas

    @V_gas.setter
    def V_gas(self, val):
        if val <= 0.0:
            raise ValueError(f"V_gas should be positive (passed {val})")
        self._V_gas = float(val)

    @property
    def T_ad(self):
        return self._T_ad

    @T_ad.setter
    def T_ad(self, val):
        if val <= 0.0:
            raise ValueError(f"T_ad should be positive (passed {val})")
        self._T_ad = float(val)

    @property
    def T_op(self):
        if self._T_op is None:
            return self._T_ad
        return self._T_op

    @T_op.setter
    def T_op(self, val):
        if val is None:
            self._T_op = None
            return None
        if val <= 0.0:
            raise ValueError(f"T_op should be positive (passed {val})")
        self._T_op = float(val)

    def save(self, path):
        """Save DigesterInformation object to .json file"""
        with open(path, "w", encoding="UTF-8") as file:
            json.dump(
                {
                    "V_liq": self.V_liq,
                    "V_gas": self.V_gas,
                    "T_ad": self.T_ad,
                    "T_op": self.T_op,
                },
                file,
            )

    def __str__(self):
        return str.join(
            "\n",
            [
                f"V_liq: {self.V_liq}",
                f"V_gas: {self.V_gas}",
                f"T_ad: {self.T_ad}",
                f"T_op: {self.T_op}",
            ],
        )

    def __repr__(self):
        return str.join(
            "\n",
            [
                f"V_liq: {self.V_liq}",
                f"V_gas: {self.V_gas}",
                f"T_ad: {self.T_ad}",
                f"T_op: {self.T_op}",
            ],
        )


def load_dig_info(path: str) -> DigesterInformation:
    with open(path, "r", encoding="UTF-8") as file:
        dig_info = json.load(file)

    if not "T_op" in dig_info.keys():
        dig_info["T_op"] = dig_info["T_ad"]

    return DigesterInformation(
        V_liq=dig_info["V_liq"],
        V_gas=dig_info["V_gas"],
        T_ad=dig_info["T_ad"],
        T_op=dig_info["T_op"],
    )
