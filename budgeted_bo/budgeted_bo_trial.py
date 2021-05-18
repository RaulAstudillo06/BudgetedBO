#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional

import logging
import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import ExpectedImprovement
from torch import Tensor

from acquisition_functions.budgeted_multi_step_ei import BudgetedMultiStepExpectedImprovement
from acquisition_functions.ei_puc import ExpectedImprovementPerUnitOfCost
from utils import (
    evaluate_obj_and_cost_at_X, 
    fit_model,
    generate_initial_design, 
    get_suggested_budget, 
    optimize_acqf_and_get_suggested_point
)

def budgeted_bo_trial(
    problem: str,
    algo: str,
    algo_params: Optional[Dict],
    trial: int,
    restart: bool,
    objective_function: Optional[Callable],
    cost_function: Optional[Callable],
    objective_cost_function: Optional[Callable],
    input_dim: int,
    n_init_evals: int,
    budget: float,
    n_max_iter: int = 200
) -> None:
    # Make sure that objective and cost functions are passed
    if (objective_cost_function is None) and (objective_function is None or cost_function is None):
        raise RuntimeError(
            "Both the objective and cost functions must be passed as inputs.")

    # Modify algo's name to account for hyperparameters
    if algo == "B-MS-EI":
        algo_id = algo + "_"
        for n in algo_params.get("lookahead_n_fantasies"):
            algo_id += str(n)

        algo_id += "_"

        if algo_params.get("refill_until_lower_bound_is_reached"):
            algo_id += "1"
        else:
            algo_id += "0"

        algo_id += "_"

        if algo_params.get("soft_plus_transform_budget"):
            algo_id += "1"
        else:
            algo_id += "0"
    else:
        algo_id = algo

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = project_path + "/experiments_results/" + problem + "/" + algo_id + "/"

    if restart:
        # Check if training data is already available
        try:
            # Current available evaluations
            X = torch.tensor(np.loadtxt(results_folder +
                                        "X/X_" + str(trial) + ".txt"))
            objective_X = torch.tensor(np.loadtxt(
                results_folder + "objective_X/objective_X_" + str(trial) + ".txt"))
            cost_X = torch.tensor(np.loadtxt(
                results_folder + "cost_X/cost_X_" + str(trial) + ".txt"))

            # Historical best observed objective values and running times
            hist_best_obs_vals = list(np.loadtxt(
                results_folder + "best_obs_vals_" + str(trial) + ".txt"))
            running_times = list(np.loadtxt(
                results_folder + "running_times/running_times_" + str(trial) + ".txt"))

            # Current best observed objective value
            best_obs_val = torch.tensor(hist_best_obs_vals[-1])

            iteration = len(hist_best_obs_vals) - 1
            print("Restarting experiment from available data.")

        except:

            # Initial evaluations
            X = generate_initial_design(
                num_samples=n_init_evals, input_dim=input_dim, seed=trial)
            objective_X, cost_X = evaluate_obj_and_cost_at_X(
                X=X, objective_function=objective_function, cost_function=cost_function, objective_cost_function=objective_cost_function)

            # Current best objective value
            best_obs_val = objective_X.max().item()

            # Historical best observed objective values and running times
            hist_best_obs_vals = [best_obs_val]
            running_times = []

            iteration = 0
    else:
        # Initial evaluations
        X = generate_initial_design(
            num_samples=n_init_evals, input_dim=input_dim, seed=trial)
        objective_X, cost_X = evaluate_obj_and_cost_at_X(
            X=X, objective_function=objective_function, cost_function=cost_function, objective_cost_function=objective_cost_function)
        print(X)
        print(objective_X)
        print(cost_X)
        # Current best objective value
        best_obs_val = objective_X.max().item()

        # Historical best observed objective values and running times
        hist_best_obs_vals = [best_obs_val]
        running_times = []

        iteration = 0

    cumulative_cost = cost_X.sum().item()
    budget_plus_init_cost = cost_X[:n_init_evals].sum().item() + budget

    algo_params["init_budget"] = budget

    while cumulative_cost <= budget_plus_init_cost and iteration <= n_max_iter:
        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + algo_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # New suggested point
        t0 = time.time()

        new_x = get_new_suggested_point(
            algo=algo,
            X=X,
            objective_X=objective_X,
            cost_X=cost_X,
            budget_left=budget_plus_init_cost - cumulative_cost,
            algo_params=algo_params,
        )

        t1 = time.time()
        running_times.append(t1 - t0)

        # Evaluate objective at new point
        objective_new_x, cost_new_x = evaluate_obj_and_cost_at_X(
            X=new_x, objective_function=objective_function, cost_function=cost_function, objective_cost_function=objective_cost_function)

        # Update training data
        X = torch.cat([X, new_x], 0)
        objective_X = torch.cat([objective_X, objective_new_x], 0)
        cost_X = torch.cat([cost_X, cost_new_x], 0)

        # Update historical best observed objective values and cumulative cost
        cumulative_cost = cost_X.sum().item()
        best_obs_val = objective_X.max().item()
        hist_best_obs_vals.append(best_obs_val)
        print("Best value found so far: " + str(best_obs_val))
        print("Remaining budget: " + str(budget_plus_init_cost - cumulative_cost))

        # Save data
        if not os.path.exists(results_folder) :
            os.makedirs(results_folder)
        if not os.path.exists(results_folder + "running_times/"):
            os.makedirs(results_folder + "running_times/")
        if not os.path.exists(results_folder + "X/"):
            os.makedirs(results_folder + "X/")
        if not os.path.exists(results_folder + "objective_X/"):
            os.makedirs(results_folder + "objective_X/")
        if not os.path.exists(results_folder + "cost_X/"):
            os.makedirs(results_folder + "cost_X/")
        np.savetxt(results_folder + "X/X_" + str(trial) + ".txt", X.numpy())
        np.savetxt(results_folder + "objective_X/objective_X_" +
                   str(trial) + ".txt", objective_X.numpy())
        np.savetxt(results_folder + "cost_X/cost_X_" +
                   str(trial) + ".txt", cost_X.numpy())
        np.savetxt(results_folder + "best_obs_vals_" +
                   str(trial) + ".txt", np.atleast_1d(hist_best_obs_vals))
        np.savetxt(results_folder + "running_times/running_times_" +
                   str(trial) + ".txt", np.atleast_1d(running_times))


def get_new_suggested_point(
    algo: str,
    X: Tensor,
    objective_X: Tensor,
    cost_X: Tensor,
    budget_left: float,
    algo_params: Optional[Dict] = None,
) -> Tensor:

    input_dim = X.shape[-1]
    algo_params["budget_left"] = budget_left

    if algo == "Random":
        return torch.rand([1, input_dim])
    elif algo == "B-MS-EI":
        # Model
        model = fit_model(
            X=X,
            objective_X=objective_X,
            cost_X=cost_X,
            training_mode="objective_and_cost",
            noiseless_obs=True,
        )

        # Acquisition function
        suggested_budget, lower_bound, fantasy_optimizers = get_suggested_budget(
            strategy="fantasy_costs_from_aux_policy",
            refill_until_lower_bound_is_reached=algo_params["refill_until_lower_bound_is_reached"],
            budget_left=budget_left,
            model=model,
            n_lookahead_steps=len(algo_params.get(
                "lookahead_n_fantasies")) + 1,
            X=X,
            objective_X=objective_X,
            cost_X=cost_X,
            init_budget=algo_params.get("init_budget"),
            previous_budget=algo_params.get("current_budget"),
            lower_bound=algo_params.get("lower_bound"),
            fantasy_optimizers=algo_params.get("aux_sequential_fantasy_costs"),
        )

        algo_params["current_budget"] = suggested_budget
        algo_params["current_budget_plus_cumulative_cost"] = suggested_budget + cost_X.sum().item()
        algo_params["lower_bound"] = lower_bound
        algo_params["aux_sequential_fantasy_costs"] = fantasy_optimizers

        if algo_params.get("soft_plus_transform_budget"):
            algo_params["beta"] = 2.0 / cost_X.min().item()
        else:
            algo_params["beta"] = None

        acquisition_function = BudgetedMultiStepExpectedImprovement(
            model=model,
            budget_plus_cumulative_cost=algo_params.get("current_budget_plus_cumulative_cost"),
            batch_size=1,
            lookahead_batch_sizes=[1 for _ in algo_params.get(
                "lookahead_n_fantasies")],
            num_fantasies=algo_params.get("lookahead_n_fantasies"),
            soft_plus_transform_budget=algo_params.get(
                "soft_plus_transform_budget"),
            beta=algo_params.get("beta"),
        )
    elif algo == "EI":
        # Model
        model = fit_model(
            X=X,
            objective_X=objective_X,
            cost_X=cost_X,
            training_mode="objective",
            noiseless_obs=True,
        )

        acquisition_function = ExpectedImprovement(
            model=model, best_f=objective_X.max().item())
            
    elif algo == "EI-PUC":
        # Model
        model = fit_model(
            X=X,
            objective_X=objective_X,
            cost_X=cost_X,
            training_mode="objective_and_cost",
            noiseless_obs=True,
        )

        # Acquisition function
        acquisition_function = ExpectedImprovementPerUnitOfCost(
            model=model,
            best_f=objective_X.max().item(),
        )

    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])

    new_x = optimize_acqf_and_get_suggested_point(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=1,
        algo_params=algo_params,
    )

    return new_x
