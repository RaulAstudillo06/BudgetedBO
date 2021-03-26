#!/usr/bin/env python3

import logging
import time
import warnings
from copy import copy
from typing import Any, Dict, List

import fblearner.flow.api as flow

# @dep=//pytorch/botorch_fb:botorch_fb
import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.objective import ScalarizedObjective
from botorch.acquisition.utils import (
    expand_trace_observations,
    project_to_target_fidelity,
)
from botorch.exceptions.warnings import BadInitialCandidatesWarning
from botorch.generation.gen import get_best_candidates
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor

from .benchmark_utils import (
    CA_ALGOS_LIST,
    MF_ALGOS_LIST,
    OneSampleRun,
    TAlgoParams,
    gen_candidates,
    generate_initial_data,
    get_acquisition_function,
    get_best_point,
    make_new_observation,
    torch_to_list,
)
from .budgeted_bo_ultis import (
    get_lagrange_multiplier,
    get_mean_costs,
    get_suggested_budget,
    initialize_model,
)
from .problems import get_problem


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)


@flow.flow_async(resource_requirements=flow.ResourceRequirements(cpu=4, memory="16g"))
@flow.typed()
def run_one_ca_trial_cpu(
    seed: int,
    problem: str,
    budget: float,
    is_log_cost: bool,
    algos: List[str],
    algo_params: List[TAlgoParams],
    general_params: Dict[str, int],
    double: bool = True,
    num_trace_observations: int = 0,
    id: str = "",
) -> OneSampleRun:
    return run_one_ca_trial(
        seed=seed,
        problem=problem,
        budget=budget,
        is_log_cost=is_log_cost,
        algos=algos,
        algo_params=algo_params,
        general_params=general_params,
        cuda=False,
        double=double,
        num_trace_observations=num_trace_observations,
        id=id,
    )


@flow.flow_async(
    resource_requirements=flow.ResourceRequirements(cpu=3, gpu=1, memory="10g")
)
@flow.typed()
def run_one_ca_trial_cuda(
    seed: int,
    problem: str,
    budget: float,
    is_log_cost: bool,
    algos: List[str],
    algo_params: List[TAlgoParams],
    general_params: Dict[str, int],
    double: bool = True,
    num_trace_observations: int = 0,
    id: str = "",
) -> OneSampleRun:
    return run_one_ca_trial(
        seed=seed,
        problem=problem,
        budget=budget,
        is_log_cost=is_log_cost,
        algos=algos,
        algo_params=algo_params,
        general_params=general_params,
        cuda=True,
        double=double,
        num_trace_observations=num_trace_observations,
        id=id,
    )


def run_one_ca_trial(  # noqa C901
    seed: int,
    problem: str,
    budget: float,
    is_log_cost: bool,
    algos: List[str],
    algo_params: List[TAlgoParams],
    general_params: Dict[str, Any],
    cuda: bool = False,
    double: bool = True,
    num_trace_observations: int = 0,
    id: str = "",
) -> OneSampleRun:

    torch.manual_seed(seed)
    logging.info(f"Trial seed = {seed}")
    logging.info(f"cuda = {cuda}, double = {double}")

    if cuda and not torch.cuda.is_available():
        raise Exception("GPU is not available.")

    # get the benchmark problem (i.e., objective, noise, bounds, etc)
    problem_spec = get_problem(problem=problem, cuda=cuda, double=double, seed=seed)
    noiseless_obs = problem_spec.noise_se is None or problem_spec.noise_se == 0.0

    # define expand and project functions
    fidelity_dims = problem_spec.fidelity_dims
    if fidelity_dims is not None:
        target_fidelities = {f: 1.0 for f in fidelity_dims}

        def project(X: Tensor) -> Tensor:
            return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

        def expand(X: Tensor) -> Tensor:
            return expand_trace_observations(
                X=X, fidelity_dims=fidelity_dims, num_trace_obs=num_trace_observations
            )

    # call helper functions to generate initial training data and initialize model
    train_x, train_obj, _, train_cost = generate_initial_data(
        n=problem_spec.n_points,
        problem_spec=problem_spec,
        sobol=True,
        sobol_seed=seed.item(),
        cuda=cuda,
        double=double,
        num_trace_observations=num_trace_observations,
    )

    init_cost = train_cost.sum().item()
    budget_plus_init_cost = budget + init_cost

    train_xs = []
    train_objs = []
    train_costs = []
    mlls = []
    models = []
    cost_mlls = []
    cost_models = []
    recommended_xs = []
    func_evals = []
    cumulative_costs = []
    budgets = []
    fantasy_costs_at_suggested_xs = []
    wall_times = []

    for i, _ in enumerate(algos):
        train_xs.append(train_x)
        train_objs.append(train_obj)
        train_costs.append(train_cost)
        mlls.append(None)
        models.append(None)
        cost_mlls.append(None)
        cost_models.append(None)
        recommended_xs.append([])
        func_evals.append([])
        cumulative_costs.append([train_cost.sum().item()])
        budgets.append([])
        fantasy_costs_at_suggested_xs.append([])
        wall_times.append([])

        if fidelity_dims is not None:
            algo_params[i]["project"] = project
            algo_params[i]["fidelity_dims"] = problem_spec.fidelity_dims
            algo_params[i]["expand"] = expand

    failure_mode = "success"
    N_MAX_ITER = 250

    # run BayesOpt until budget is exhausted or the number of iterations exceeds
    for i, algo in enumerate(algos):
        n_iter = 1
        learned_cost_mode = (algo in CA_ALGOS_LIST) or (algo in MF_ALGOS_LIST)
        if learned_cost_mode:
            logging.info("Using a learned cost model.")
        else:
            logging.info("A learned cost model is not required by this algorithm.")

        joint_objective_cost_model = (algo in CA_ALGOS_LIST) and (
            algo not in ["NCA_EI", "CA_EI_cool"]
        )
        log_scale_acqf = (algo in CA_ALGOS_LIST) and (
            algo not in ["NCA_EI", "CA_EI_cool"]
        )
        cost_aware_utility = None

        while cumulative_costs[i][-1] <= budget_plus_init_cost and n_iter <= N_MAX_ITER:

            #########
            # Step 1: Fit model.
            if joint_objective_cost_model:
                train_y_model = torch.cat(
                    (train_objs[i].unsqueeze(dim=-1), train_costs[i].unsqueeze(dim=-1)),
                    dim=-1,
                )

                training_mode = "objective_and_cost"

            else:
                train_y_model = train_objs[i].clone()
                training_mode = "objective"

                if learned_cost_mode:
                    cost_mll, cost_model = initialize_model(
                        train_x=train_xs[i].clone(),
                        train_y=train_costs[i].clone(),
                        problem_spec=problem_spec,
                        noiseless_obs=noiseless_obs,
                        training_mode="cost",
                        log_transform_y=is_log_cost,
                        log_scale_acqf=log_scale_acqf,
                    )

                    cost_mlls[i] = cost_mll
                    cost_models[i] = cost_model
                    fit_gpytorch_model(cost_mlls[i])

                    cost_aware_utility = InverseCostWeightedUtility(
                        cost_model=cost_models[i], use_mean=True
                    )

                    logging.info("Using a learned cost model.")

            mll, model = initialize_model(
                train_x=train_xs[i].clone(),
                train_y=train_y_model,
                problem_spec=problem_spec,
                training_mode=training_mode,
                noiseless_obs=noiseless_obs,
                log_transform_y=is_log_cost,
                log_scale_acqf=log_scale_acqf,
            )

            mlls[i] = mll
            models[i] = model
            fit_gpytorch_model(mlls[i])

            #########
            # Step 2: Get best point so far.
            if joint_objective_cost_model:
                weights = torch.tensor(
                    [1.0, 0.0],
                    dtype=problem_spec.bounds.dtype,
                    device=problem_spec.bounds.device,
                )
                best_point_acqf_objective = ScalarizedObjective(weights=weights)
            else:
                best_point_acqf_objective = None

            new_rec_x, value = get_best_point(
                model=models[i],
                algo=algo,
                algo_params=algo_params[i],
                general_params=general_params,
                problem_spec=problem_spec,
                train_x=train_xs[i],
                train_obj=train_objs[i],
                return_best_only=False,
                objective=best_point_acqf_objective,
            )

            if algo == "NCKGF":
                kg_options = {
                    "num_trace_observations": algo_params[i]["num_trace_observations"],
                    "value_maximizer": normalize(
                        new_rec_x.clone(), bounds=problem_spec.bounds
                    ),
                    "partial_restarts": algo_params[i]["partial_restarts"],
                    "fidelity_bound": [0.0, 1.0],
                }
            elif "MS_" in algo:
                if n_iter == 1:
                    suggested_budget = None
                    lower_bound = None
                    kg_options = {"full_optimizer": None}
                else:
                    best_new_rec_x = get_best_candidates(
                        batch_candidates=new_rec_x, batch_values=value
                    )
                    kg_options["value_maximizer"] = normalize(
                        best_new_rec_x.clone(), bounds=problem_spec.bounds
                    )

            else:
                kg_options = None

            new_rec_x = get_best_candidates(
                batch_candidates=new_rec_x, batch_values=value
            )

            recommended_xs[i].append(new_rec_x.squeeze().tolist())

            if problem_spec.obj_cost_function is not None:
                obj_rec = problem_spec.obj_cost_function(new_rec_x)[0].item()
            else:
                obj_rec = problem_spec.objective(new_rec_x).item()

            func_evals[i].append(obj_rec)
            logging.info(f"    Algorithm = {algo}")
            logging.info(f"    Number of evals = {n_iter -  1}")
            logging.info(f"    Recommended point = {new_rec_x.squeeze(dim=0)}")
            logging.info(f"    Value at recommended point = {obj_rec:>4.4f}")

            #########
            # Step 3: Define the acquisition function.

            if "KG" in algo:
                algo_params[i]["current_value"] = value.max()
            else:
                algo_params[i]["current_value"] = train_objs[i].max().item()

            if algo == "CA_UCB":
                input_dim = problem_spec.dimension
                algo_params[i]["beta_obj"] = np.sqrt(
                    0.2 * input_dim * np.log(2.0 * input_dim * n_iter + 1.0)
                )
                algo_params[i]["beta_cost"] = np.sqrt(
                    0.2 * input_dim * np.log(2.0 * input_dim * n_iter + 1.0)
                )
            elif algo == "BC_MS_EI":
                cumulative_cost = train_costs[i].sum().item()
                budget_left = budget_plus_init_cost - cumulative_cost
                n_lookahead_steps = len(algo_params[i].get("lookahead_n_fantasies")) + 1
                budget_scheduling_strategy = algo_params[i].get(
                    "budget_scheduling_strategy"
                )
                refill_until_lower_bound_is_reached = algo_params[i].get(
                    "refill_until_lower_bound_is_reached"
                )
                suggested_budget, lower_bound = get_suggested_budget(
                    strategy=budget_scheduling_strategy,
                    refill_until_lower_bound_is_reached=refill_until_lower_bound_is_reached,
                    budget_left=copy(budget_left),
                    model=models[i],
                    n_lookahead_steps=n_lookahead_steps,
                    train_x=train_xs[i].clone(),
                    train_obj=train_objs[i].clone(),
                    train_cost=train_costs[i].clone(),
                    problem_spec=problem_spec,
                    init_budget=copy(budget),
                    previous_budget=suggested_budget,
                    lower_bound=lower_bound,
                )
                budgets[i].append(copy(suggested_budget))
                algo_params[i]["budget"] = cumulative_cost + suggested_budget
            elif algo == "L_MS_EI":
                n_lookahead_steps = len(algo_params[i].get("lookahead_n_fantasies")) + 1
                algo_params[i]["multiplier"] = get_lagrange_multiplier(
                    model=models[i], problem_spec=problem_spec
                )
            elif algo == "CA_EI_cool":
                cumulative_cost = train_costs[i].sum().item()
                budget_left = budget_plus_init_cost - cumulative_cost
                algo_params[i]["cost_exponent"] = budget_left / budget

            acq_func = get_acquisition_function(
                model=models[i],
                algo=algo,
                algo_params=algo_params[i],
                train_x=train_xs[i].clone(),
                train_obj=train_objs[i].clone(),
                train_cost=train_costs[i].clone(),
                cost_model=cost_models[i],
                cost_aware_utility=cost_aware_utility,
                is_log_cost=is_log_cost,
                bounds=problem_spec.bounds,
                winsorization_level=problem_spec.winsorization_level,
            )

            #########
            # Step 4: Optimize the acquisition function and get candidates.
            t0 = time.time()
            try:
                if algo == "CA_TS":
                    input_dim = problem_spec.dimension
                    discretization_size = (
                        algo_params[i]["discretization_size_per_dim"] * input_dim
                    )
                    domain_discretization = torch.rand(
                        discretization_size, dtype=problem_spec.bounds.dtype
                    ).view(algo_params[i]["discretization_size_per_dim"], input_dim)
                    new_x = acq_func(X=domain_discretization)
                    new_x = unnormalize(new_x, bounds=problem_spec.bounds)
                else:
                    new_x, acq_value = gen_candidates(
                        acq_func=acq_func,
                        algo=algo,
                        algo_params=algo_params[i],
                        general_params=general_params,
                        problem_spec=problem_spec,
                        cuda=cuda,
                        double=double,
                        kg_options=kg_options,
                        return_best_only=True,
                    )
            except Exception as e:
                if not general_params["catch_errors"]:
                    raise e
                logging.info(f"Failed to optimize in {algo} with error: {e}")
                logging.info("Generating point uniformly at random.")
                new_x, acq_value = gen_candidates(
                    acq_func=acq_func,
                    algo="RND",
                    algo_params=algo_params[i],
                    general_params=general_params,
                    problem_spec=problem_spec,
                    cuda=cuda,
                    double=double,
                    kg_options=kg_options,
                    return_best_only=True,
                )

            if algo == "BC_MS_EI":
                # get fantasy obs at suggested x
                fantasy_model_at_x_opt = acq_func.get_induced_fantasy_model(
                    X=kg_options["suggested_x_full_tree"].unsqueeze(0)
                )
                fantasy_costs_at_x_opt = get_mean_costs(
                    model=fantasy_model_at_x_opt,
                    n_observations=len(algo_params[i].get("lookahead_n_fantasies")),
                )
                fantasy_costs_at_suggested_xs[i].append(
                    fantasy_costs_at_x_opt.squeeze()
                )

            # track wall time
            wall_times[i].append(time.time() - t0)
            logging.info(f"    Opt time = {wall_times[i][-1]:>4.4f}s")

            #########
            # Step 5. Evaluate the function and make new observation.
            logging.info(f"    Suggested point = {new_x.squeeze(dim=0)}")
            new_obj, _, new_cost = make_new_observation(
                new_x=new_x, problem_spec=problem_spec
            )

            n_iter += 1

            #########
            # Step 6. Update training data.
            train_xs[i] = torch.cat((train_xs[i], new_x))
            train_objs[i] = torch.cat((train_objs[i], new_obj))
            train_costs[i] = torch.cat((train_costs[i], new_cost))

            # assumes that aggregation of q-batch costs is to take the sum
            cumulative_costs[i].append(train_costs[i].sum().item())
            logging.info(
                f"    Cumulative cost = {cumulative_costs[i][-1] - init_cost:>4.4f}"
            )

        fantasy_costs_at_suggested_xs[i] = torch_to_list(
            fantasy_costs_at_suggested_xs[i]
        )

        if failure_mode != "success":
            break

    #########
    train_objs = [train_obj.unsqueeze(-1) for train_obj in train_objs]

    algos_adjusted = []
    for i, algo in enumerate(algos):
        if "MS_EI" in algo:
            n_lookahead_fantasies = algo_params[i].get("lookahead_n_fantasies")
            n_lookahead_steps = len(n_lookahead_fantasies) + 1
            algo = str(n_lookahead_steps) + "-" + algo + "-"
            for n in n_lookahead_fantasies:
                algo += str(n)
            if algo_params[i].get("refill_until_lower_bound_is_reached"):
                algo += "rulbr"
        algos_adjusted.append(algo)

    return OneSampleRun(
        algos=[id + "-" + a for a in algos_adjusted],
        func_evals=func_evals,
        recs=recommended_xs,
        train_xs=torch_to_list(train_xs),
        train_objs=torch_to_list(train_objs),
        train_costs=torch_to_list(train_costs),
        train_cons=[[]],
        wall_times=wall_times,
        cost_evals=[[]],
        acq_value_vec=[[]],
        failure_mode=failure_mode,
        hvs=[[]],
        stopping_information=[[]],
        budgets=budgets,
        fantasy_costs=fantasy_costs_at_suggested_xs,
    )
