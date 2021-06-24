import random
import torch

from botorch import acquisition
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead, warmstart_multistep
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import IIDNormalSampler
from copy import deepcopy
from torch import Tensor
from typing import List, Dict

from budgeted_bo.acquisition_functions.budgeted_ei import BudgetedExpectedImprovement
from budgeted_bo.acquisition_functions.budgeted_multi_step_ei import BudgetedMultiStepExpectedImprovement


def custom_warmstart_multistep(
    acq_function: qMultiStepLookahead,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    full_optimizer: Tensor,
    algo_params: Dict,
) -> Tensor:

    batch_initial_conditions = warmstart_multistep(
            acq_function=acq_function,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            full_optimizer=full_optimizer,
        )
    
    n_initial_points = batch_initial_conditions.shape[0]
    random_index = random.randrange(n_initial_points)
    print(random_index)
    input_dim = batch_initial_conditions.shape[-1]
    batch_shape, shapes, sizes = acq_function.get_split_shapes(X=batch_initial_conditions)

    # Safe copy of model
    model = deepcopy(acq_function.model)
    obj_model = model.subset_output(idcs=[0])

    #Define optimization domain
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    
    #
    aux_acq_func = PosteriorMean(model=obj_model)

    new_x, acq_value = optimize_acqf(
        acq_function=aux_acq_func,
        bounds=standard_bounds,
        q=1,
        num_restarts=5 * input_dim,
        raw_samples=100 * input_dim,
        options={
            "batch_limit": 5,
            "maxiter": 100,
            "method": "L-BFGS-B",
        },
        return_best_only=True,
    )

    i = 0
    for _ in range(sizes[0]):
        batch_initial_conditions[random_index, i, :] = new_x.clone().squeeze(0)
        i += 1

    # Fantasize objective and cost values
    sampler = IIDNormalSampler(
        num_samples=1, resample=True, collapse_batch_dims=True
    )
    posterior_new_x = model.posterior(new_x, observation_noise=True)
    fantasy_obs = sampler(posterior_new_x).squeeze(dim=0).detach()
    fantasy_cost = torch.exp(fantasy_obs[0,1]).item()
    model = model.condition_on_observations(X=new_x, Y=fantasy_obs)

    n_lookahead_steps = len(algo_params.get("lookahead_n_fantasies")) - 1

    if n_lookahead_steps > 0:
        aux_acq_func = BudgetedMultiStepExpectedImprovement(
            model=model,
            budget_plus_cumulative_cost=algo_params.get("current_budget_plus_cumulative_cost") - fantasy_cost,
            batch_size=1,
            lookahead_batch_sizes=[1 for _ in range(n_lookahead_steps)],
            num_fantasies=[1 for _ in range(n_lookahead_steps)],
            soft_plus_transform_budget=algo_params.get(
                "soft_plus_transform_budget"),
            beta=algo_params.get("beta"),
        )
    else:
        y = torch.transpose(model.train_targets, -2, -1)
        y_original_scale = model.outcome_transform.untransform(y)[0]
        obj_vals = y_original_scale[..., 0]
        best_f = torch.max(obj_vals).item()

        aux_acq_func = BudgetedExpectedImprovement(
            model=model,
            best_f=best_f,
            budget=algo_params.get("current_budget") - fantasy_cost,
        )


    new_x, acq_value = optimize_acqf(
            acq_function=aux_acq_func,
            bounds=standard_bounds,
            q=aux_acq_func.get_augmented_q_batch_size(1) if n_lookahead_steps > 0 else 1,
            num_restarts=5 * input_dim,
            raw_samples=100 * input_dim,
            options={
                "batch_limit": 5,
                "maxiter": 100,
                "method": "L-BFGS-B",
            },
            return_best_only=True,
            return_full_tree=True,
        )

    for j, size in enumerate(sizes[1:]):
        for _ in range(size):
            batch_initial_conditions[random_index, i, :] = new_x[j, :].clone()
            i += 1

    return batch_initial_conditions
    