import os
import sys
import numpy as np
import torch

from botorch.settings import debug
from botorch.test_functions.synthetic import Ackley
from math import pi
from typing import Callable
from torch import Tensor
torch.set_default_dtype(torch.float64)
debug._set_state(True)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:-12])

from budgeted_bo.experiment_manager import experiment_manager

from sklearn_problem_gen import sklearn_classifier_objective


# Objective and cost functions
def get_objective_cost_function(seed: int) -> Callable:

    dataset_name = "Boston"
    model_name = "RandomForestReg"


    def objective_cost_function(X):
        X_unscaled = X.clone()
        X_unscaled[:, 0] = 255.0 * X_unscaled[:, 0] + 1.0
        X_unscaled[:, 1] = 63.0 * X_unscaled[:, 1] + 1.0
        X_unscaled[:, 2] = X_unscaled[:, 2] - 1.0
        return sklearn_classifier_objective(
            X=X_unscaled, dataset_name=dataset_name, model_name=model_name
        )

    return [objective_cost_function]


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
    problem="rfboston",
    algo=algo,
    algo_params=algo_params,
    restart=False,
    first_trial=first_trial,
    last_trial=last_trial,
    get_objective_cost_function=get_objective_cost_function,
    input_dim=3,
    n_init_evals=8,
    budget=10.0,
)
