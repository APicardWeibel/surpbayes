"""
Class for a single task in a Meta Learning environnement

Note:
A task instance does not have all the information required to perform the training. Notably, it
lacks the ProbaMap object necessary to interpret the post_param, as well as the prior. As such,
the task can only be interpreted in the wider context of a MetaLearningEnv object.
"""

import os
from typing import Callable, Optional

import dill
import numpy as np
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.load_accu import load_accu_sample_val
from surpbayes.types import MetaData, ProbaParam, SamplePoint


class Task:
    """
    Class for a single learning task.

    A task instance does not have all the information required to perform the training. Notably, it
    lacks the ProbaMap object necessary to interpret the post_param, as well as the prior. As such,
    the task can only be interpreted in the wider context of a MetaLearningEnv object.

    Attributes:
        score: scoring function for the task.
        temperature: the temperature of the task.
        post_param: the currently trained posterior parameter (prior as well as ProbaMap is
            specified in the MetaLearningEnv).
        accu_sample_val: AccuSampleVal container for evaluation of the score.
        meta_data: MetaData of the task.
        save_path: where the task data is saved.
        parallel: should 'score' calls be parallelized.
        vectorized: is 'score' vectorized.

    Method:
        save. Save the task data (can be loaded later through load_task function).
            score and meta_data are pickled (using 'dill' package). The rest are stored in
            human readable function.
    """

    def __init__(
        self,
        score: Callable[[SamplePoint], float],
        temperature: float = 1.0,
        post_param: Optional[ProbaParam] = None,
        accu_sample_val: Optional[AccuSampleVal] = None,
        meta_data: Optional[MetaData] = None,
        save_path: Optional[str] = None,
        parallel: bool = True,
        vectorized: bool = False,
    ):
        self.score = score
        self.post_param = post_param
        self.accu_sample_val = accu_sample_val
        self.temp = temperature

        self.meta_data = meta_data

        self.parallel = parallel
        self.vectorized = vectorized

        self.end_score = None
        self.save_path = save_path

    def save(
        self, name: Optional[str], path: Optional[str] = None, overwrite: bool = True
    ) -> str:
        """Save Task data

        Task data is saved in folder 'name' situated at 'path'.
        If 'name' is not provided, defaults to saving in 'save_path'.

        Function "score" is saved using 'dill' library.
        """
        # Check that path + name provided are coherent
        if name is not None:
            if path is None:
                path = "."
            if not os.path.isdir(path):
                raise ValueError(f"{path} should point to a folder")
            acc_path = os.path.join(path, name)
            os.makedirs(acc_path, exist_ok=overwrite)
        else:
            acc_path = self.save_path  # type: ignore
            if acc_path is None:
                raise ValueError(
                    "Could not interpret where to save (save_path argument missing and name not provided)"
                )

        # Save post_param
        if self.post_param is not None:
            np.savetxt(os.path.join(acc_path, "post_param.csv"), self.post_param)

        # Save temp
        with open(os.path.join(acc_path, "temp.txt"), "w", encoding="utf-8") as file:
            file.write(str(self.temp))

        # Save accu_sample_val
        # The name of the subclass is also saved for future loading
        if self.accu_sample_val is not None:
            self.accu_sample_val.save(
                name="accu_sample_val", path=acc_path, overwrite=overwrite
            )

        # Save score function (dill)
        with open(os.path.join(acc_path, "score.dl"), "wb") as file:
            dill.dump(self.score, file)

        # Save meta_data (dill)
        with open(os.path.join(acc_path, "meta_data.dl"), "wb") as file:
            dill.dump(self.meta_data, file)

        return acc_path


def load_task(path) -> Task:
    """
    Load a task from a folder.

    Note
    The type of accu sample used is read from 'accu_type.txt' file.
    """
    with open(os.path.join(path, "score.dl"), "rb") as file:
        score = dill.load(file)
    with open(os.path.join(path, "temp.txt"), "r", encoding="utf-8") as file:
        temp = float(file.read())

    if os.path.isfile(os.path.join(path, "post_param.csv")):
        post_param = np.loadtxt(os.path.join(path, "post_param.csv"))
    else:
        post_param = None

    if os.path.isdir(os.path.join(path, "accu_sample_val")):
        accu_sample_val = load_accu_sample_val(os.path.join(path, "accu_sample_val"))
    else:
        accu_sample_val = None

    with open(os.path.join(path, "meta_data.dl"), "rb") as file:
        meta_data = dill.load(file)

    return Task(
        score=score,
        temperature=temp,
        post_param=post_param,
        accu_sample_val=accu_sample_val,
        meta_data=meta_data,
        save_path=path,
    )
