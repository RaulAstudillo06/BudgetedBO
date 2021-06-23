import os
import sys
import numpy as np
import torch

from botorch.settings import debug
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
        beta = torch.tensor([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]) / 10.0
        C_t = torch.tensor(
            [
                [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            ],
            dtype=torch.float,
        )
        C = C_t.transpose(-1, -2)

        X_unnorm = X * 10.0

        objective_X = sum(
            1 / (torch.sum((X_unnorm - C[i]) ** 2, dim=-1) + beta[i])
            for i in range(5)
        )
        return objective_X


    def cost_function(X: Tensor) -> Tensor:
        X_unnorm = X * 10.0
        a, b, c = get_cost_function_parameters(seed=seed % 20)
        ln_cost_X = a * torch.cos(b * (2.0 * pi / 4.0) * (X_unnorm - 4.0 + c)).mean(dim=-1)
        cost_X = torch.exp(ln_cost_X)
        return cost_X

    return [objective_function, cost_function]


# Algos
algo = "EI-PUC-CC"

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
    problem="shekel5",
    algo=algo,
    algo_params=algo_params,
    restart=False,
    first_trial=first_trial,
    last_trial=last_trial,
    get_objective_cost_function=get_objective_cost_function,
    input_dim=4,
    n_init_evals=10,
    budget=80.0,
)
