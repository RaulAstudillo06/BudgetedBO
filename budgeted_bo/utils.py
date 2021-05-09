
import logging
from copy import copy
from typing import Callable, Dict, Optional

import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead, warmstart_multistep
from botorch.acquisition.objective import ScalarizedObjective
from botorch.generation.gen import get_best_candidates
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.model import Model
from botorch.models.transforms.outcome import (
    ChainedOutcomeTransform,
    Log,
    Standardize,
)
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.samplers import IIDNormalSampler
from botorch.utils.transforms import normalize
from acquisition_functions.naive_cost_aware import (
    NaiveCostAwareAcquisitionFunction,
)
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


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
    cost_model: Optional[Model] = None,
):
    """
    Fantasizes the observed costs when following a specified sampling
    policy for a given number of steps.
    """
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])

    if algo == "CA_EI_cool":
        for _ in range(n_steps):
            obj_model = model
            standardized_obj_vals = obj_model.train_targets
            obj_vals = obj_model.outcome_transform.untransform(standardized_obj_vals)[0]
            best_f = torch.max(obj_vals).item()

            raw_aux_acq_func = ExpectedImprovement(model=obj_model, best_f=best_f)

            # Baseline acquisition fucntion
            cost_exponent = budget_left / init_budget

            aux_acq_func = NaiveCostAwareAcquisitionFunction(
                raw_acqf=raw_aux_acq_func,
                cost_model=cost_model,
                use_mean=True,
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

            # Fantasize objective and cost values
            obj_sampler = IIDNormalSampler(
                num_samples=1, resample=True, collapse_batch_dims=True
            )
            obj_post_X = obj_model.posterior(new_x, observation_noise=True)
            fantasy_obj_val = obj_sampler(obj_post_X).squeeze(dim=0)
            obj_model = obj_model.condition_on_observations(X=new_x, Y=fantasy_obj_val)

            cost_sampler = IIDNormalSampler(
                num_samples=1, resample=True, collapse_batch_dims=True
            )
            cost_post_X = cost_model.posterior(new_x, observation_noise=True)
            fantasy_cost_val = cost_sampler(cost_post_X).squeeze(dim=0)
            cost_model = cost_model.condition_on_observations(
                X=new_x, Y=fantasy_cost_val
            )

            # Update remaining budget
            transformed_costs = cost_model.train_targets
            costs = cost_model.outcome_transform.untransform(transformed_costs)[0]
            budget_left = budget_left - costs.squeeze()[-1].item()
        transformed_costs = cost_model.train_targets
        costs = cost_model.outcome_transform.untransform(transformed_costs)[0]
        fantasy_costs = costs.squeeze()[-n_steps:].detach()

    logging.info("Fantasy costs:")
    logging.info(fantasy_costs)
    return fantasy_costs


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
        Y = torch.log(cost_X)
    elif training_mode == "objective_and_cost":
        Y = torch.cat((objective_X.unsqueeze(dim=-1), torch.log(cost_X).unsqueeze(dim=-1)), dim=-1)

    
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)

    # Outcome transform
    outcome_transform = Standardize(m=Y.shape[-1], batch_shape=Y.shape[:-2])

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


def generate_initial_design(num_samples: int, input_dim: int, seed=None):
    # generate training data
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
        return suggested_budget, lower_bound

    if strategy == "fantasy_costs_from_aux_policy":
        # Get objective and cost models (Note that, since we originally model
        # the log cost, the cost model cannot be obtained by subsetting the
        # original model)
        obj_model = model.subset_output(idcs=[0])

        cost_model = fit_model(
            X=X,
            objective_X=objective_X,
            cost_X=cost_X,
            training_mode="cost",
            noiseless_obs=True,
        )

        # Fantasize the observed costs following the auxiliary acquisition function
        fantasy_costs = fantasize_costs(
            algo="CA_EI_cool",
            model=obj_model,
            n_steps=n_lookahead_steps,
            budget_left=copy(budget_left),
            init_budget=init_budget,
            input_dim=X.shape[-1],
            cost_model=cost_model,
        )

        # Suggested budget is the minimum between the sum of the fantasy costs
        # and the true remaining budget.
        suggested_budget = fantasy_costs.sum().item()
        lower_bound = fantasy_costs.min().item()
    suggested_budget = min(suggested_budget, budget_left)
    return suggested_budget, lower_bound


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
    raw_samples = 100 * input_dim * batch_size
    num_restarts =  5 * input_dim * batch_size

    if False:#is_ms:
        raw_samples *= (len(algo_params.get("lookahead_n_fantasies")) + 1)
        num_restarts *=  (len(algo_params.get("lookahead_n_fantasies")) + 1)

    if algo_params.get("suggested_x_full_tree") is not None:
        batch_initial_conditions = warmstart_multistep(
            acq_function=acq_func,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            full_optimizer=algo_params.get("suggested_x_full_tree"),
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
            return_best_only=not is_ms,
            return_full_tree=is_ms,
        )

    candidates =  candidates.detach()
    if is_ms:
        # save all tree variables for multi-step initialization
        algo_params["suggested_x_full_tree"] = candidates.clone()
        candidates = acq_func.extract_candidates(candidates)

    print(acq_values.shape)
    print(candidates.shape)
    acq_values_sorted, indices = torch.sort(acq_values.squeeze()) 
    print(acq_values_sorted)
    #print(candidates)

    new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
    print(new_x.shape)
    print(error)
    return new_x
