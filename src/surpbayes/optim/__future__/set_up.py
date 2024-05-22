"""Module for interface between optim module and users"""

from typing import Callable

import numpy as np


class OptimProblem:
    def __init__(
        self,
        x0: dict[str, float],
        lower_bounds: dict[str, float],
        upper_bounds: dict[str, float],
        fun: Callable,
        all_pos: bool = False,
        **kwargs,
    ):
        self.x0 = list(x0.values())
        self.param_names = param_names = list(x0.keys())

        # Check that no names in bounds is not accounted for
        if not set(lower_bounds.keys()).issubset(param_names):
            unknown_ = set(lower_bounds.keys()).difference(param_names)
            raise ValueError(
                f"lower_bounds contains entries missing from param_names ({unknown_})"
            )
        if not set(upper_bounds.keys()).issubset(param_names):
            unknown_ = set(lower_bounds.keys()).difference(param_names)
            raise ValueError(
                f"upper_bounds contains entries missing from param_names ({unknown_})"
            )

        # Complete bounds
        missing_names = set(param_names).difference(upper_bounds.keys())
        self.upper_bounds = upper_bounds.copy()
        self.upper_bounds.update({x: np.inf for x in missing_names})
        self.act_upb = {x: np.isinf(self.upper_bounds[x]) for x in param_names}

        lower_limit = 0 if all_pos else -np.inf
        missing_names = set(param_names).difference(lower_bounds.keys())
        self.lower_bounds = lower_bounds.copy()
        self.lower_bounds.update({x: lower_limit for x in missing_names})
        self.act_lob = {x: np.isinf(self.lower_bounds[x]) for x in param_names}

        self.no_act = np.array(
            [(not self.act_lob[x]) and (not self.act_upb[x]) for x in param_names]
        )
        self.only_l = np.array(
            [self.act_lob[x] and (not self.act_upb[x]) for x in param_names]
        )
        self.only_u = np.array(
            [(not self.act_lob[x]) and self.act_upb[x] for x in param_names]
        )
        self.both_ul = np.array(
            [self.act_lob[x] and self.act_upb[x] for x in param_names]
        )
        self.lob_arr = np.array([lower_bounds[x] for x in param_names])
        self.upb_arr = np.array([upper_bounds[x] for x in param_names])

        self.span_both = (self.upb_arr - self.lob_arr)[self.both_ul]

        if all_pos:
            if not all([x >= 0.0 for x in lower_bounds.values()]):
                raise ValueError(
                    "'lower_bounds' values should all be positive if 'all_pos' is True"
                )

        self.n_param = len(param_names)

        self.fun = fun
        self.kwargs = kwargs
        self.x0_free = self.convert_dict(x0)

    def convert(self, t) -> dict[str, float]:
        """t is drawn in np.ndarray format and lives in R. Force bounds on t"""
        x = t.copy()

        # for upper and lower limit, use a tanh transform
        x[self.both_ul] = (
            self.lob_arr[self.both_ul]
            + self.span_both * (np.tanh(t[self.both_ul]) + 1) / 2
        )

        # for the rest, use an exponential transform
        x[self.only_l] = self.lob_arr[self.only_l] + (
            self.x0[self.only_l] - self.lob_arr[self.only_l]
        ) * np.exp(t)
        x[self.only_u] = self.upb_arr[self.only_u] - (
            self.upb_arr[self.only_u] - self.x0[self.only_u]
        ) * np.exp(-t)

        return dict(zip(self.param_names, x))

    def convert_dict(self, dico):
        num_arr = np.array([dico[x] for x in self.param_names])
        num_arr[self.both_ul] = np.arctanh(
            2 * (num_arr[self.both_ul] - self.lob_arr[self.both_ul]) / self.span_both
            - 1
        )
        num_arr[self.only_l] = np.log(
            (num_arr[self.only_l] - self.lob_arr[self.only_l])
            / (self.x0[self.only_l] - self.lob_arr[self.only_l])
        )
        num_arr[self.only_u] = -np.log(
            -(num_arr[self.only_u] - self.lob_arr[self.only_u])
            / (self.upb_arr[self.only_u] - self.x0[self.only_u])
        )
        return num_arr

    def eval(self, t, **kwargs) -> float:
        """Fun function with array input"""
        return self.fun(**self.convert(t), **kwargs)

    def optimize(self):
        """Basically pass to CMA ES routine"""
        pass

    def numpy_to_dict(self, xs):
        return dict(zip(self.param_names, xs))

    def loc_fun(self, xs: np.ndarray, **kwargs):
        return self.fun(**self.numpy_to_dict(xs), **kwargs)
