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
from utils import fit_model


# Objective and cost functions
def get_objective_cost_function(seed: int) -> Callable:
    data = scipy.io.loadmat("lda_on_grid.mat")
    data = data["lda_on_grid"]


    X = torch.tensor(data[:, :3])
    X[:, 0] = 2.0 * X[:, 0] - 1.0
    X[:, 1] = torch.log2(X[:, 1]) / 10.0
    X[:, 2] = torch.log2(X[:, 2]) / 14.0

    objective_X = -torch.tensor(data[:, 3]).unsqueeze(-1)
    cost_X = torch.tensor(data[:, 4]).unsqueeze(-1) / 3600.0
    log_cost_X = torch.log(cost_X)

    objective_model_state_dict = torch.load('lda_objective_model_state.pth')
    objective_model = fit_model(
        X=X,
        objective_X=objective_X,
        cost_X=cost_X,
        training_mode="objective",
        noiseless_obs=True,
    )
    objective_model.load_state_dict(objective_model_state_dict)

    log_cost_model_state_dict = torch.load('lda_log_cost_model_state.pth')
    log_cost_model = fit_model(
        X=X,
        objective_X=log_cost_X,
        cost_X=cost_X,
        training_mode="objective",
        noiseless_obs=True,
    )
    log_cost_model.load_state_dict(log_cost_model_state_dict)
    
    def objective_function(X: Tensor) -> Tensor:
        posterior_X = objective_model.posterior(X)
        objective_X = posterior_X.mean.squeeze(-1).detach()
        return objective_X


    def cost_function(X: Tensor) -> Tensor:
        posterior_X = log_cost_model.posterior(X)
        cost_X = torch.exp(posterior_X.mean + 0.5 * posterior_X.variance).squeeze(-1).detach()
        return cost_X

    return [objective_function, cost_function]


# Algos
algo = "EI-PUC"

if algo == "B-MS-EI": 
    algo_params = {"lookahead_n_fantasies": [4, 4, 2], "refill_until_lower_bound_is_reached": True, "soft_plus_transform_budget": True}
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
    problem="lda",
    algo=algo,
    algo_params=algo_params,
    restart=False,
    first_trial=first_trial,
    last_trial=last_trial,
    get_objective_cost_function=get_objective_cost_function,
    input_dim=3,
    n_init_evals=8,
    budget=100.0,
)