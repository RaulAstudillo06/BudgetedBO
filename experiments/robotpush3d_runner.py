import os
import sys
import numpy as np
import scipy.io
import torch

from botorch.settings import debug
from botorch.models import SingleTaskGP
from math import pi
from typing import Callable
from torch import Tensor
torch.set_default_dtype(torch.float64)
debug._set_state(True)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:-11] + "budgeted_bo")
sys.path.append(script_dir[:-11] + "budgeted_bo/acquisition_functions")

from experiment_manager import experiment_manager
from robot_pushing_src.robot_pushing_3d import robot_pushing_3d
from utils import fit_model


# Objective and cost functions
def get_objective_cost_function(seed: int) -> Callable:
    original_state = torch.random.get_rng_state()
    torch.manual_seed(seed % 20)
    tartget_location = 10.0 * torch.rand([2]) - 5.0
    #avoid_location = 10.0 * torch.rand([2]) - 5.0
    #avoid_location = tartget_location + 2.0 * torch.rand([2]) - 1.0
    #avoid_location = torch.clamp(avoid_location, min=-5.0, max=5.0)    
    torch.random.set_rng_state(original_state)

    
    def objective_cost_function(X: Tensor) -> Tensor:
        X_unnorm = X.clone()
        X_unnorm[:, :2] = 10.0 * X_unnorm[:, :2] - 5.0
        X_unnorm[:, 2] = 29.0 * X_unnorm[:, 2] + 1.0
        objective_X = []
        cost_X = []
        for x in X_unnorm:
            np.random.seed(0)
            object_location = torch.tensor(robot_pushing_3d(x[0].item(), x[1].item(), x[2].item()))
            objective_X.append(-torch.dist(tartget_location, object_location))
            cost_X.append(torch.norm(object_location))
        np.random.seed()
        objective_X = torch.tensor(objective_X)
        cost_X = torch.tensor(cost_X)
        return objective_X, cost_X


    return [objective_cost_function]


# Algos
algo = "EI"

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
    problem="robotpush3d",
    algo=algo,
    algo_params=algo_params,
    restart=False,
    first_trial=first_trial,
    last_trial=last_trial,
    get_objective_cost_function=get_objective_cost_function,
    input_dim=3,
    n_init_evals=8,
    budget=200.0,
)