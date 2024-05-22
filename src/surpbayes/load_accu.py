import os
import warnings

# Import all types of AccuSampleVal child
from surpbayes.accu_xy import AccuSampleVal
from surpbayes.basic_io import safe_load_json
from surpbayes.bayes import AccuSampleValDens, AccuSampleValExp

# accu_type interpretation
# This should be exhaustive
accu_type_to_class = {
    "AccuSampleVal": AccuSampleVal,
    "AccuSampleValDens": AccuSampleValDens,
    "AccuSampleValExp": AccuSampleValExp,
}


# Loading minimum shape information for initialisation of the Accu.
def mini_loader_std(path) -> dict:
    path_sample_shape = os.path.join(path, "sample_shape.json")

    sample_shape = tuple(safe_load_json(path_sample_shape, encoding="utf-8"))
    return {"sample_shape": sample_shape}


def mini_loader_exp(path) -> dict:
    path_sample_shape = os.path.join(path, "sample_shape.json")
    path_t_shape = os.path.join(path, "t_shape.json")

    sample_shape = tuple(safe_load_json(path_sample_shape, encoding="utf-8"))
    t_shape = tuple(safe_load_json(path_t_shape, encoding="utf-8"))

    return {"sample_shape": sample_shape, "t_shape": t_shape}


minimum_loader = {
    "AccuSampleVal": mini_loader_std,
    "AccuSampleValDens": mini_loader_std,
    "AccuSampleValExp": mini_loader_exp,
}

if not set(minimum_loader.keys()) == set(accu_type_to_class.keys()):
    raise ValueError(
        " ".join(
            [
                "Implementation error: mismatch between minimum_loader",
                "and accu_type_to_class",
                "(probably missing some accu subclass)",
            ]
        )
    )


def load_accu_sample_val(path: str) -> AccuSampleVal:
    """
    Load an object inherited from AccuSampleVal from a folder.

    The specific type of the object is inferred from 'accu_type.txt' file.
    """

    accu_type_path = os.path.join(path, "acc_type.txt")
    with open(accu_type_path, "r", encoding="utf-8") as file:
        accu_type_name = file.read()

    if accu_type_name not in accu_type_to_class:
        warnings.warn(
            f"Unrecognized accu_type '{accu_type_name}'. Defaulting to 'AccuSampleVal'"
        )
        accu_type_name = "AccuSampleVal"

    shape_dict_info = minimum_loader[accu_type_name](path)
    acc = accu_type_to_class[accu_type_name](n_tot=1, **shape_dict_info)

    acc.load(path)
    return acc
