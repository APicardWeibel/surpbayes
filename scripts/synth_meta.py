# pylint: disable=all
import argparse
import os
from math import pi
from time import time

import numpy as np
from scipy.stats import ortho_group
from surpbayes.meta_bayes import MetaLearningEnv, Task
from surpbayes.meta_bayes.test_assess import eval_meta_hist
from surpbayes.proba import GaussianMap

loc_path, _ = os.path.split(__file__)

data_path = os.path.join(loc_path, "..", "ADM1_data_LF")
save_path = os.path.join(loc_path, "..", "exp_results", "meta_learning")
os.makedirs(save_path, exist_ok=True)

# Create synthetic empirical risk function
# The resulting function takes values in 0/1
def perturb(x, omega):
    return (np.cos(x * omega) - 1.0)/omega + x

def make_task(
    x:np.ndarray,
    half_hess:np.ndarray,
    omega:float= 2*pi,
    temperature:float=0.1,
    ):

    def score(xs:np.ndarray):
        mod_delta = (x- xs) @ half_hess
        
        return np.tanh(perturb((mod_delta ** 2).sum(-1), omega) / 10)
    
    return Task(score, temperature=temperature, vectorized=True)

# Define a score function generator
# Care is taken to ensure that the score is not quadratic and is bounded
# However, it remains nearly quadratic close to the minima (as for all
# C2 function though) and has a single local minima.
class TaskGen:
    """Class for generating tasks
    
    """
    def __init__(
        self,
        temp: float,
        mean_param:np.ndarray,
        half_draw_cov:np.ndarray,
        pre_hess:np.ndarray,
        omega_min:float= 3/2 * pi,
        omega_max:float=5/2 * pi,
        hess_sd:float = 0.1
    ):
        self.__temp = temp
        
        self.__mpar = np.asarray(mean_param)
        self.__hdc = np.asarray(half_draw_cov)
        self.__dim = len(self.__mpar)
        assert self.__hdc.shape == (self.__dim, self.__dim)
        
        self.__mhhess = np.asarray(pre_hess)
        assert self.__mhhess.shape == (self.__dim, self.__dim)
        
        self.__omega_min = omega_min
        self.__omega_max = omega_max
        self.__hess_sd = hess_sd
        

    def draw_params(self, n:int):
        return self.__mpar + np.random.normal(0,1, (n, self.__dim)) @ self.__hdc
    
    def draw_hhess_s(self, n:int):
        return self.__mhhess + np.random.normal(0,self.__hess_sd, (n, self.__dim, self.__dim))

    def draw_omegas(self,n:int):
        return np.random.uniform(self.__omega_min, self.__omega_max, n)
    
    
    def draw_tasks(self, n)->tuple[list[Task], np.ndarray, np.ndarray, np.ndarray]:
        params = self.draw_params(n)
        hhess_s = self.draw_hhess_s(n)
        omegas = self.draw_omegas(n)
        
        return [make_task(param, hhess, omega, self.__temp) for param, hhess, omega in zip(params, hhess_s, omegas)], params, hhess_s, omegas

def main(temp:float, d:int=8, true_dim:int=2, n_train_tasks:int=100, eps_false_dim:float=0.05, n_test_tasks:int=40, hess_sd:float=0.1):

    ortho_mat = ortho_group(d).rvs(1)

    half_draw_cov = np.diag(np.concatenate([np.exp(np.random.uniform(-0.5, 0.5, true_dim)), np.full(d-true_dim, eps_false_dim)])) @ ortho_mat

    pre_hess = np.eye(d)

    mean_param = np.random.normal(0, 1, d)
    mean_param = 2.0 * mean_param / np.sqrt(np.sum(mean_param ** 2)) # Force norm of 2

    temp_str = str(temp).replace('.', '_')
    np.savetxt(os.path.join(save_path, f"mean_param_{temp_str}.csv"), mean_param)
    np.savetxt(os.path.join(save_path, f"half_draw_cov_{temp_str}.csv"), half_draw_cov)

    task_gen = TaskGen(
        temp=temp,
        mean_param=mean_param,
        half_draw_cov=half_draw_cov,
        pre_hess=pre_hess,
        hess_sd=hess_sd)

    # Check that accurate number of dimensions is chosen
    assert np.sum((np.linalg.eigvalsh(np.cov(task_gen.draw_params(10000), rowvar=False))) > (eps_false_dim**2) * 1.5) == true_dim

    train_tasks, params, hhess_s, omegas = task_gen.draw_tasks(n_train_tasks)

    proba_map = GaussianMap(d)

    mlearn = MetaLearningEnv(
        proba_map,
        list_task=train_tasks,
        per_step=[100] * 5 + [50] * 10, # Initial 
        chain_length=15,
        kl_max=0.5,
        dampen=0.7,
        n_estim_weights=10**4,
        silent=True,
    )

    tic = time()
    mlearn.meta_learn(epochs=1, eta=1/temp, kl_max=0.2, mini_batch_size=10) # Initial training
    tac = time()
    print("-" * 10, f"Complete initial training in {tac-tic} s", "-" * 10)

    # Change hyperparameters
    mlearn.hyperparams.update({"per_step":[20, 0] * 2, "chain_length": 4, "dampen": 0.3})
    tic = time()
    mlearn.meta_learn(epochs = 19, eta = 1.0/temp, kl_max = 0.2, mini_batch_size=20)
    mlearn.meta_learn(epochs = 30, eta = 0.5/temp, kl_max = 0.1, mini_batch_size=20)
    mlearn.meta_learn(epochs = 100, eta = 0.4/temp, kl_max = 0.1, mini_batch_size=20)
    tac = time()
    print("-" * 10, f"Complete second training phase in {tac-tic} s", "-" * 10)

    # Change hyperparameters for last phase
    mlearn.hyperparams.update({"per_step":[10, 0, ] * 2})
    tic = time()
    mlearn.meta_learn_batch(epochs=50, eta = 0.2/temp, kl_max = 0.1, kl_tol=10**-4)
    tac = time()
    print("-" * 10, f"Complete last training phase in {tac-tic} s", "-" * 10)

    np.savetxt(f"train_perfs_{temp_str}.csv", mlearn.hist_meta.meta_scores())
    test_tasks, _, _, _ = task_gen.draw_tasks(n_test_tasks)

    eval_index = list(range(10)) + list(range(10, mlearn.hist_meta.n_filled, 5))
    test_perf = eval_meta_hist(
        mlearn.hist_meta.meta_params()[eval_index],
        test_tasks, proba_map = proba_map,
        n_test = 10**4,
        hyperparams= {"per_step":[100] * 5 + [50] * 15, "chain_length": 20})

    np.savetxt(os.path.join(save_path, f"perfs_{temp_str}.csv"), test_perf)

if __name__ == "__main__":
    # evaluate two temperatures: 0.1 and 0.01
    main(temp=0.1)
    main(temp=0.01)
