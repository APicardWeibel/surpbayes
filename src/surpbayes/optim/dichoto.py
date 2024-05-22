"""
Dichotomy solver for fun(x) = y, assuming fun is monotonous.

If fun is not monotonous, solver should also find a solution, though it might not be unique.
"""
import warnings
from math import ceil
from typing import Callable, Optional

import numpy as np


class SolverFailed(Exception):
    """Exception class for solver failure"""


def dichoto(
    fun: Callable[[float], float],
    y: float,
    x_min: float = 0.0,
    x_max: Optional[float] = None,
    increasing: bool = True,
    y_pres: Optional[float] = None,
    x_pres: Optional[float] = None,
    m_max: int = 1000,
) -> tuple[float, float]:
    """
    Dichotomy solver for equation 'fun(x) = y', assuming 'fun' is monotonous

    If 'fun' is not monotonous, the solver should find a solution to the equation,
    though one can not assume it is the only one.

    Args:
        fun: a function of R to R
        y: float, for which fun(x) = y is solved
        x_min, x_max: float, bounds for solution x.
        increasing: bool, specify if the function is assumed to be increasing or decreasing
        y_pres: precision of solver in y (condition |f(x_estim) - y| < y_pres)
        x_pres: precision of solver in x (condition |x_max - x_min| < x_pres)
        m_max: maximum number of function call.

    Return:
        a tuple x_min, x_max bracketing x such that f(x) = y.

    Raise:
        SolverFailed if solution is not found.
    """

    # Convert decreasing to increasing if needed
    mult = (-1) ** (increasing + 1)
    y = y * mult

    compt = 0

    # Look for an interval [x_min, x_max] such that y \in [f(x_min), f(x_max)].
    if x_max is None:
        # Initialize first guess for x_max
        if x_min < 0:
            # Check if x_max = 0 is suitable
            compt += 1
            if mult * fun(0) > y:
                x_max = 0.0
                x_max_routine = False
            else:
                # 0 is not large enough. Start guess with - x_min
                x_max = -x_min
                x_min = 0.0
                x_max_routine = True

        elif x_min == 0.0:
            x_max = 1.0
            x_max_routine = True

        else:
            x_max = 2.0 * x_min
            x_max_routine = True

        # If x_max routine, then it is necessary to increase x_max. It is doubled at each step
        if x_max_routine:
            while (mult * fun(x_max) < y) & (compt < m_max):
                x_min = x_max
                x_max = 2.0 * x_max
                compt += 1

            # Failed to find a suitable x_max
            if compt >= m_max:
                raise SolverFailed(
                    f"Could not find an interval containing {y}. Last couple: {(x_min, x_max)}"
                )
    # If x_max is provided, check that [f(x_min), f(x_max)] contains y
    else:
        # Check that interval contains y
        f_min = mult * fun(x_min)
        f_max = mult * fun(x_max)
        if (y < f_min) or (y > f_max):
            raise SolverFailed(
                f"Interval (f(x_min), f(x_max)) does not contain {mult * y}"
            )

    # Pre compute max precision in x achievable
    max_pres = 2.0 ** (-(m_max - compt))

    if x_pres is None:
        n_iter = m_max - compt
        x_pres = max(10 ** (-3) * (x_max - x_min), 2 ** (-n_iter))
    elif max_pres > x_pres:
        warnings.warn("The required precision in x could not be achieved")
        return (x_min, x_max)
    else:
        n_iter = ceil(np.log2((x_max - x_min) / x_pres))  # Number of precisions

    # First dichotomy loop: until x criteria is met
    for _ in range(n_iter):
        x_new = (x_max + x_min) / 2.0
        f_new = mult * fun(x_new)
        if f_new >= y:
            x_max = x_new
        else:
            x_min = x_new
        compt += 1

    # Now enforce precision on y.
    if y_pres is None:
        return (x_min, x_max)
    else:
        err_min = y - mult * fun(x_min)
        err_max = mult * fun(x_max) - y

        while (compt < m_max) & (max(err_min, err_max) > y_pres):
            x_new = (x_max + x_min) / 2.0
            f_new = mult * fun(x_new)
            if f_new >= y:
                x_max = x_new
                err_max = f_new - y
            else:
                x_min = x_new
                err_min = f_new  # TO DO: DOUBLE CHECK THAT LINE!!

            compt += 1
        if compt >= m_max:
            warnings.warn("The required precision in y could not be achieved")
        return (x_min, x_max)
