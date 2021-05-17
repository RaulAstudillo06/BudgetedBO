
import logging
from copy import copy, deepcopy
from typing import Callable, Dict, Optional

import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead, warmstart_multistep
from botorch.acquisition.objective import ScalarizedObjective
from botorch.generation.gen import get_best_candidates
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.outcome import (
    ChainedOutcomeTransform,
    Log,
    Standardize,
)
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.samplers import IIDNormalSampler
from botorch.utils.transforms import normalize
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor

from acquisition_functions.ei_puc import ExpectedImprovementPerUnitOfCost
from custom_warmstart_multistep import custom_warmstart_multistep


def evaluate_obj_and_cost_at_X(
    X: Tensor,
    objective_function: Optional[Callable],
    cost_function: Optional[Callable],
    objective_cost_function: Optional[Callable],
)-> Tensor:
    if (objective_cost_function is None) and (objective_function is None or cost_function is None):
        raise RuntimeError(
            "Both the objective and cost functions must be passed as inputs.")
    
    if objective_cost_function is not None:
        objective_X, cost_X = objective_cost_function(X)
    else:
        objective_X = objective_function(X)
        cost_X = cost_function(X)
    
    return objective_X, cost_X


def fantasize_costs(
    algo: str,
    model: Model,
    n_steps: int,
    budget_left: float,
    init_budget: float,
    input_dim: int,
):
    """
    Fantasizes the observed costs when following a specified sampling
    policy for a given number of steps.
    """
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])

    fantasy_costs = []
    fantasy_optimizers = []

    if algo == "EI-PUC_CC":
        for _ in range(n_steps):
            # Acquisition function
            y = torch.transpose(model.train_targets, -2, -1)
            y_original_scale = model.outcome_transform.untransform(y)[0]
            obj_vals = y_original_scale[..., 0]
            best_f = torch.max(obj_vals).item()
            cost_exponent = budget_left / init_budget

            aux_acq_func = ExpectedImprovementPerUnitOfCost(
                model=model,
                best_f=best_f,
                cost_exponent=cost_exponent,
            )

            # Get new point
            new_x, acq_value = optimize_acqf(
                acq_function=aux_acq_func,
                bounds=standard_bounds,
                q=1,
                num_restarts=5 * input_dim,
                raw_samples=50 * input_dim,
                options={
                    "batch_limit": 5,
                    "maxiter": 100,
                    "nonnegative": True,
                    "method": "L-BFGS-B",
                },
                return_best_only=True,
            )

            fantasy_optimizers.append(new_x.clone())

            # Fantasize objective and cost values
            sampler = IIDNormalSampler(
                num_samples=1, resample=True, collapse_batch_dims=True
            )
            posterior_new_x = model.posterior(new_x, observation_noise=True)
            fantasy_obs = sampler(posterior_new_x).squeeze(dim=0).detach()
            fantasy_costs.append(torch.exp(fantasy_obs[0,1]).item())
            model = model.condition_on_observations(X=new_x, Y=fantasy_obs)

            # Update remaining budget
            budget_left -= fantasy_costs[-1]

    print("Fantasy costs:")
    fantasy_costs = torch.tensor(fantasy_costs)
    print(fantasy_costs)
    return fantasy_costs, fantasy_optimizers


def fit_model(
    X: Tensor,
    objective_X: Tensor,
    cost_X: Tensor,
    training_mode: str,
    noiseless_obs: bool = False,
):
    if training_mode == "objective":
        Y = objective_X
    elif training_mode == "cost":
        Y = cost_X
    elif training_mode == "objective_and_cost":
        Y = torch.cat((objective_X.unsqueeze(dim=-1), torch.log(cost_X).unsqueeze(dim=-1)), dim=-1)

    
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)

    # Outcome transform
    standardize = Standardize(m=Y.shape[-1], batch_shape=Y.shape[:-2])
    if training_mode == "cost":
        log = Log()
        outcome_transform = ChainedOutcomeTransform(
                log=log, standardize=standardize
            )
    else:
        outcome_transform = standardize

    # Likelihood
    if noiseless_obs:
        _, aug_batch_shape = SingleTaskGP.get_batch_dimensions(
            train_X=X,
            train_Y=Y,
        )
        likelihood = GaussianLikelihood(
            batch_shape=aug_batch_shape,
            noise_constraint=Interval(lower_bound=1e-4, upper_bound=1e-3),
        )
    else:
        likelihood = None

    # Define model
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        likelihood=likelihood,
        outcome_transform=outcome_transform,
    )

    model.outcome_transform.eval()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    return model


def generate_initial_design(num_samples: int, input_dim: int, sobol: bool=True, seed: int=None):
    # generate training data
    if sobol:
        soboleng = torch.quasirandom.SobolEngine(dimension=input_dim, scramble=True, seed=seed)
        X = soboleng.draw(num_samples).to(dtype=torch.double)
    else:
        if seed is not None:
            old_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
            X = torch.rand([num_samples, input_dim])
            torch.random.set_rng_state(old_state)
        else:
            X = torch.rand([num_samples, input_dim])
    return X


def get_suggested_budget(
    strategy: str,
    refill_until_lower_bound_is_reached: bool,
    budget_left: float,
    model: Model,
    n_lookahead_steps: int,
    X: Tensor,
    objective_X: Tensor,
    cost_X: Tensor,
    init_budget: float,
    previous_budget: Optional[float] = None,
    lower_bound: Optional[float] = None,
    fantasy_optimizers: Optional[float] = None,
):
    """
    Computes the suggested budget to be used by the budgeted multi-step
    expected improvement acquisition function.
    """
    if (
        refill_until_lower_bound_is_reached
        and (lower_bound is not None)
        and (previous_budget - cost_X[-1] > lower_bound)
    ):
        suggested_budget = previous_budget - cost_X[-1].item()
        return suggested_budget, lower_bound, fantasy_optimizers

    if strategy == "fantasy_costs_from_aux_policy":

        # Fantasize the observed costs following the auxiliary acquisition function
        fantasy_costs, fantasy_optimizers = fantasize_costs(
            algo="EI-PUC_CC",
            model=deepcopy(model),
            n_steps=n_lookahead_steps,
            budget_left=copy(budget_left),
            init_budget=init_budget,
            input_dim=X.shape[-1],
        )

        # Suggested budget is the minimum between the sum of the fantasy costs
        # and the true remaining budget.
        suggested_budget = fantasy_costs.sum().item()
        lower_bound = fantasy_costs.min().item()
    suggested_budget = min(suggested_budget, budget_left)
    return suggested_budget, lower_bound, fantasy_optimizers


def optimize_acqf_and_get_suggested_point(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    algo_params: Dict,
) -> Tensor:
    """Optimizes the acquisition function, and returns the candidate solution."""
    is_ms = isinstance(acq_func, qMultiStepLookahead)
    input_dim = bounds.shape[1]
    q = acq_func.get_augmented_q_batch_size(batch_size) if is_ms else batch_size
    raw_samples = 200 * input_dim * batch_size
    num_restarts =  10 * input_dim * batch_size

    #if is_ms:
        #raw_samples *= (len(algo_params.get("lookahead_n_fantasies")) + 1)
        #num_restarts *=  (len(algo_params.get("lookahead_n_fantasies")) + 1)

    if algo_params.get("suggested_x_full_tree") is not None:
        batch_initial_conditions = custom_warmstart_multistep(
            acq_function=acq_func,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            full_optimizer=algo_params.get("suggested_x_full_tree"),
            algo_params=algo_params,
        )
    else:
        batch_initial_conditions = None

    candidates, acq_values = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={
                "batch_limit": 5,
                "maxiter": 100,
                "nonnegative": True,
                "method": "L-BFGS-B",
            },
            batch_initial_conditions=batch_initial_conditions,
            return_best_only=False,
            return_full_tree=is_ms,
        )

    candidates =  candidates.detach()
    if is_ms:
        # save all tree variables for multi-step initialization
        algo_params["suggested_x_full_tree"] = candidates.clone()
        candidates = acq_func.extract_candidates(candidates)

    acq_values_sorted, indices = torch.sort(acq_values.squeeze(), descending=True)
    print("Acquisition values:") 
    print(acq_values_sorted)
    print("Candidates:")
    print(candidates[indices].squeeze())

    new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
    return new_x
