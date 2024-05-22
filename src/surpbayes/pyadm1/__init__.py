"""
PyADM1 - Anaerobic digestion modelisation with Anerobic Digestion Model 1 (ADM1)

Details about the modelisation can be found at https://doi.org/10.2166/wst.2002.0292 .
This package is based around ADM1 implementation from https://github.com/CaptainFerMag/PyADM1 .

PyADM1 is designed for:

    - Modelisation (run_adm1)
    - Sensitivity Analysis (submodule SA)
    - Calibration (submodule optim)
    - Uncertainty Quantification (submodule UQ)

Permanent data storage is organised around submodule IO.
Anaerobic Digestion objects (DigesterInfo, DigesterFeed, DigesterParameter, DigesterState,
DigesterStates) are stored as human readable files (csv, json) and can be loaded/saved using
load_(dig_feed/dig_info/dig_param/dig_state/dig_states) and save methods.

Main functions:
    run_adm1 (ADM1 modeling of digester from initial state, digester information, digester
        parameter and influent)
    adm1_err (measures the difference between predictions and observations)
    score_param (measures the difference between predictions using a specific parameters and
        observations)
    adm1_derivative (computes the derivative of ADM1 with respect to the digester parameter)


(Licence:
Copyright (c) 2021 Peyman Sadrimajd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
)
"""

from . import basic_classes
from .prediction_error import adm1_err
from .proba import prior_param, proba_map
