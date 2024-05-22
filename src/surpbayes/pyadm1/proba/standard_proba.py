"""
Standard distributions and distribution maps for Variational inference and other Bayesian methods.

Distributions are defined on the Free Digester Parameter space (i.e. log space).
The transform maps 0 to the default digester parameter, so that the means of default distributions
 should be 0.

Default standard deviations for each parameter are defined in the _normalisation_param.py file in
the IO submodule.
"""

import numpy as np
# from surpbayes.proba import (BlockDiagGaussMap, FixedCovGaussianMap, GaussianMap,
#                         TensorizedGaussianMap)
from surpbayes.proba import BlockDiagGaussMap
from surpbayes.pyadm1.basic_classes.parameter import ADM1Param
from surpbayes.pyadm1.proba._normalisation_param import (devs_dict, devs_interp,
                                                    renorm_param)

# ------- Group diagonal covariance - Indexes to train -------
parameter_families = [
    ["k_dis", "k_hyd_ch", "k_hyd_pr", "k_hyd_li"],
    ["k_m_su", "K_S_su"],
    ["k_m_aa", "K_S_aa"],
    ["pH_UL:LL_aa", "pH_LL_aa"],
    ["k_m_fa", "K_S_fa", "K_I_h2_fa"],
    ["k_m_c4", "K_S_c4", "K_I_h2_c4"],
    ["k_m_pro", "K_S_pro", "K_I_h2_pro"],
    ["k_m_ac", "K_S_ac"],
    ["pH_UL:LL_ac", "pH_LL_ac", "K_I_nh3"],
    ["k_m_h2", "K_S_h2"],
    ["pH_UL:LL_h2", "pH_LL_h2"],
    ["k_dec"],
    ["K_S_IN"],
]

param_names = [name for names in parameter_families for name in names]
group_index = []
_count = 0
for index in parameter_families:
    k = len(index)
    group_index.append(list(range(_count, _count+k)))
    _count += k
# group_index = [[parameter_dict[name] for name in family] for family in parameter_families]

def _make_block_prior_param(group):
    """Construct prior paramter on block"""
    n = len(group)
    _prior_param = np.zeros((n+1, n))
    _prior_param[0] = np.log([renorm_param[name] for name in group])
    _prior_param[1:] = np.diag(0.4 * np.log(np.array(
        [devs_interp[devs_dict[name]] for name in group]) +1.0))
    return _prior_param.flatten()

prior_param = np.concatenate([_make_block_prior_param(group) for group in parameter_families])
proba_map = BlockDiagGaussMap(group_index)

def convert_param(param:np.ndarray)->ADM1Param:
    """Convert parameter from array to ADM1Param format"""
    pre_dict =  dict(zip(param_names, np.exp(param)))
    
    for name in ["ac", "h2", "aa"]:
        delta = pre_dict.pop(f"pH_UL:LL_{name}")
        pre_dict[f"pH_UL_{name}"] = pre_dict[f"pH_LL_{name}"] + delta

    return ADM1Param(pre_dict)


# # ------- Standard full gaussian distribution -------
# proba_param_map = GaussianMap(n_param)
# ref_proba_param = np.zeros((n_param + 1, n_param))
# ref_proba_param[1:] = np.diag(std_devs)

# ref_proba = proba_param_map(ref_proba_param)

# # Use the covariance matrix for optimisation
# default_proposal_cov = ref_proba.cov  # type: ignore

# # ------- Standard gaussian distribution with block diagonal covariance -------
# proba_param_map = BlockDiagGaussMap


# # ------- Standard diagonal gaussian distribution -------
# proba_param_t_map = TensorizedGaussianMap(sample_size=n_param)
# ref_t_proba_param = np.zeros((2, n_param))
# ref_t_proba_param[1] = std_devs

# ref_t_proba = proba_param_t_map(ref_t_proba_param)

# # ------- Standard fixed covariance gaussian distribution -------
# proba_param_fcov_map = FixedCovGaussianMap(
#     sample_size=n_param, cov=np.diag(std_devs**2)
# )
# ref_fcov_param = np.zeros(n_param)

# # Check that no parameter is forgotten
# assert np.sum([len(g) for g in parameter_families]) == len(parameter_dict)


# def group_to_proba_index(group):
#     indexes = [parameter_dict[param] for param in group]
#     accu = indexes.copy()
#     while len(indexes) > 0:
#         k = indexes.pop()
#         accu.append(n_param + n_param * k + k)
#         for j in indexes:
#             accu.append(n_param + n_param * k + j)
#             accu.append(n_param + n_param * j + k)
#     return accu


# proba_param_indexes = []
# for g in parameter_families:
#     proba_param_indexes += group_to_proba_index(g)
# proba_param_indexes.sort()
