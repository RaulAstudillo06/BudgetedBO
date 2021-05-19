import os
import sys
import numpy as np
import pickle
import torch

from botorch.settings import debug
from copy import copy
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

    with open(script_dir + '/rf_surrogate_cnn.pkl', 'rb') as obj_pickle_file:
        objective_surrogate = pickle.load(obj_pickle_file)

    with open(script_dir + '/rf_cost_surrogate_cnn.pkl', 'rb') as cost_pickle_file:
        cost_surrogate = pickle.load(cost_pickle_file)

    
    def objective_function(X: Tensor) -> Tensor:
        batch = X.dim() > 1
        if batch:
            xs = [x.detach().cpu().numpy() for x in X]
        else:
            xs = [X.detach().cpu().numpy()]

        vals = []

        for x in xs:
            test_point = np.concatenate((x[None, :], np.array([[40]])), axis=1)
            vals.append(objective_surrogate.predict(test_point)[0])

        # HPOLib convention is to minimize, so return negative objectives
        objective_X = -torch.tensor(vals).to(X)
        return objective_X


    def cost_function(X: Tensor) -> Tensor:
        batch = X.dim() > 1
        if batch:
            xs = [x.detach().cpu().numpy() for x in X]
        else:
            xs = [X.detach().cpu().numpy()]

        vals = []

        for x in xs:
            cx = np.cumsum(
                [
                    cost_surrogate.predict(
                        np.concatenate((x[None, :], np.array([[i]])), axis=1)
                    )[0]
                    for i in range(1, 41)
                ]
            )[0]
            
            vals.append(copy(cx))

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
    problem="cnn",
    algo=algo,
    algo_params=algo_params,
    restart=False,
    first_trial=first_trial,
    last_trial=last_trial,
    get_objective_cost_function=get_objective_cost_function,
    input_dim=5,
    n_init_evals=12,
    budget=10.0,
)