import os
import sys
import numpy as np
import torch

from botorch.settings import debug
from botorch.test_functions.synthetic import DropWave
from math import pi
from typing import Callable
from torch import Tensor
torch.set_default_dtype(torch.float64)
debug._set_state(True)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:-11] + "budgeted_bo")
sys.path.append(script_dir[:-11] + "budgeted_bo/acquisition_functions")

from experiment_manager import experiment_manager
from parametric_cost_function import get_cost_function_parameters


# Objective and cost functions
def get_objective_cost_function(seed: int) -> Callable:
    
    def objective_function(X: Tensor) -> Tensor:
        X_unnorm = 20 * X - 10.0 
        objective_X = -torch.abs(X_unnorm * torch.sin(X_unnorm) + 0.1 * X_unnorm).sum(dim=-1)
        return objective_X


    def cost_function(X: Tensor) -> Tensor:
        X_unnorm = 20 * X - 10.0 
        a, b, c = get_cost_function_parameters(seed=seed)
        ln_cost_X = a * torch.cos(b * (2 * pi / 10.0) * (X_unnorm + c)).mean(dim=-1)
        cost_X = torch.exp(ln_cost_X)
        return cost_X

    return [objective_function, cost_function]

# Algos
algo = "B-MS-EI"

if algo == "B-MS-EI": 
    algo_params = {"lookahead_n_fantasies": [1, 1, 1], "refill_until_lower_bound_is_reached": True, "soft_plus_transform_budget": False}
else:
    algo_params = {}

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[1])

experiment_manager(
    problem="alpine1",
    algo=algo,
    algo_params=algo_params,
    restart=False,
    first_trial=first_trial,
    last_trial=last_trial,
    get_objective_cost_function=get_objective_cost_function,
    input_dim=3,
    n_init_evals=8,
    budget=60.0,
)
