from typing import Callable, Optional, Union

import numpy as np
from multiprocess import cpu_count
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.optim.optimizer import IterOptimizer


class GenOptimizer(IterOptimizer):
    """Subclass of Optimizer for routines which at each step evaluate a generation of points

    This class is not directly usable ('update' method should be rewritten)
    """

    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        param_ini: np.ndarray,
        # Termination criteria
        chain_length: int = 100,
        xtol: float = 10 ** (-8),
        ftol: float = 10 ** (-8),
        delta_conv_check: int = 1,
        # Generation size
        per_step: Optional[Union[int, list[int]]] = None,
        prev_eval: Optional[AccuSampleVal] = None,
        # Function evaluation
        parallel: bool = True,
        vectorized: bool = False,
        # Printing
        silent=False,
        print_rec: int = 5,
        **kwargs,
    ):
        """
        Added arguments:
            per_step: number of evaluations during each update
            prev_eval: previous evaluations of fun as an AccuSampleVal
        """
        super().__init__(
            fun=fun,
            param_ini=param_ini,
            chain_length=chain_length,
            xtol=xtol,
            ftol=ftol,
            delta_conv_check=delta_conv_check,
            parallel=parallel,
            vectorized=vectorized,
            silent=silent,
            print_rec=print_rec,
            **kwargs,
        )

        self.par_shape = self.param.shape
        self.dim_par = self.param.size

        self.set_up_per_step(per_step=per_step)

        self.set_up_accu(prev_eval=prev_eval)

    def set_up_per_step(self, per_step: Optional[Union[int, list[int]]]) -> None:
        def opt_par_n(n_do, n_cpu):
            if n_do % n_cpu != 0:
                n_do = (n_do // n_cpu + 1) * n_cpu
            return n_do

        if per_step is None:
            n_target = min(50, 2 * self.dim_par)
            if self.parallel:
                n_cpu = cpu_count()
                n_target = opt_par_n(n_target, n_cpu)

            self.per_step = np.full(self.chain_length, n_target)

        elif isinstance(per_step, int):
            if self.parallel:
                n_cpu = cpu_count()
                per_step = opt_par_n(per_step, n_cpu)
            self.per_step = [per_step for _ in range(self.chain_length)]
        else:
            self.per_step = [opt_par_n(x, n_cpu) for x in per_step]

    def set_up_accu(self, prev_eval: Optional[AccuSampleVal]) -> None:
        if prev_eval is None:
            self.accu = AccuSampleVal(
                sample_shape=self.par_shape, n_tot=sum(self.per_step) + 1
            )
        else:
            self.accu = prev_eval
            self.accu.extend_memory(sum(self.per_step))
