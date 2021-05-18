import os
import sys
import numpy as np
import pickle
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
from utils import fit_model


# Objective and cost functions
def get_objective_cost_function(seed: int) -> Callable:

    with open(script_dir + '/rf_surrogate_svm.pkl', 'rb') as pickle_file:
        objective_surrogate = pickle.load(pickle_file)

    with open(script_dir + '/rf_cost_surrogate_svm.pkl', 'rb') as pickle_file:
        cost_surrogate = pickle.load(pickle_file)

    
    def objective_function(X: Tensor) -> Tensor:
        X_unnorm = 20.0 * X - 10.0
        batch = X_unnorm.dim() > 1
        if batch:
            xs = [x.detach().cpu().numpy() for x in X_unnorm]
        else:
            xs = [X_unnorm.detach().cpu().numpy()]

        vals = []

        for x in xs:
            test_point = np.append(x, 4000)[None, :]
            vals.append(objective_surrogate.predict(test_point)[0])

        # HPOLib convention is to minimize, so return negative objectives
        objective_X = -torch.tensor(vals).to(X)
        return objective_X


    def cost_function(X: Tensor) -> Tensor:
        X_unnorm = 20.0 * X - 10.0
        batch = X.dim() > 1
        if batch:
            xs = [x.detach().cpu().numpy() for x in X_unnorm]
        else:
            xs = [X_unnorm.detach().cpu().numpy()]

        vals = []

        for x in xs:
            test_point = np.append(x, 4000)[None, :]
            vals.append(cost_surrogate.predict(test_point)[0])

        cost_X = torch.tensor(vals).to(X) / 3600.0
        return cost_X


    return [objective_function, cost_function]


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
    problem="svm",
    algo=algo,
    algo_params=algo_params,
    restart=False,
    first_trial=first_trial,
    last_trial=last_trial,
    get_objective_cost_function=get_objective_cost_function,
    input_dim=2,
    n_init_evals=6,
    budget=12.0,
)
