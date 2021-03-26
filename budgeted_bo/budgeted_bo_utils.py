#!/usr/bin/env python3

import logging
from copy import copy
from typing import Optional

import torch
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.objective import ScalarizedObjective
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
from botorch_fb.experimental.acquisition.naive_cost_aware import (
    NaiveCostAwareAcquisitionFunction,
)
from fblearner.flow.projects.botorch.problems import ProblemSpec
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


def get_suggested_budget(
    strategy: str,
    refill_until_lower_bound_is_reached: bool,
    budget_left: float,
    model: Model,
    n_lookahead_steps: int,
    train_x: Tensor,
    train_obj: Tensor,
    train_cost: Tensor,
    problem_spec: ProblemSpec,
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
        and (previous_budget - train_cost[-1] > lower_bound)
    ):
        suggested_budget = previous_budget - train_cost[-1].item()
        return suggested_budget, lower_bound

    if strategy == "fantasy_costs_from_aux_policy":
        # Get objective and cost models (Note that, since we originally model
        # the log cost, the cost model cannot be obtained by subsetting the
        # original model)
        obj_model = model.subset_output(idcs=[0])

        cost_mll, cost_model = initialize_model(
            train_x=train_x,
            train_y=train_cost,
            problem_spec=problem_spec,
            noiseless_obs=True,
            training_mode="cost",
            log_transform_y=True,
            log_scale_acqf=False,
        )
        fit_gpytorch_model(cost_mll)

        # Fantasize the observed costs following the auxiliary acquisition function
        fantasy_costs = fantasize_costs(
            algo="CA_EI_cool",
            model=obj_model,
            n_steps=n_lookahead_steps,
            problem_spec=problem_spec,
            budget_left=copy(budget_left),
            init_budget=init_budget,
            cost_model=cost_model,
        )

        # Suggested budget is the minimum between the sum of the fantasy costs
        # and the true remaining budget.
        suggested_budget = fantasy_costs.sum().item()
        lower_bound = fantasy_costs.min().item()
    suggested_budget = min(suggested_budget, budget_left)
    return suggested_budget, lower_bound


def fantasize_costs(
    algo: str,
    model: Model,
    n_steps: int,
    problem_spec: ProblemSpec,
    budget_left: float,
    init_budget: float,
    cost_model: Optional[Model] = None,
):
    """
    Fantasizes the observed costs when following a specified sampling
    policy for a given number of steps.
    """
    input_dim = problem_spec.dimension
    bounds = problem_spec.bounds
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim]).to(bounds)

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


def get_mean_costs(model: Model, n_observations: int):
    """
    Extract the mean of the last `n_observations` costs within the model,
    where the mean is computed with respect to the batch dimension of the model.
    """
    y = torch.transpose(model.train_targets, -2, -1)
    y_original_scale = model.outcome_transform.untransform(y)[0]
    costs = torch.exp(y_original_scale[..., 1])
    costs = costs[..., -n_observations:]
    mean_costs = costs.mean(dim=[i for i in range(len(costs.shape) - 1)])
    return mean_costs


def get_lagrange_multiplier(model: Model, problem_spec: ProblemSpec):
    """
    Computes the Lagrange multiplier to used by the Lagrangian multi-step
    expected improvement acquisition function.
    """
    input_dim = problem_spec.dimension
    bounds = problem_spec.bounds
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim]).to(bounds)

    y = torch.transpose(model.train_targets, -2, -1)
    y_original_scale = model.outcome_transform.untransform(y)[0]
    obj_vals = y_original_scale[..., 0]
    best_f = obj_vals.max().detach()

    weights = torch.tensor(
        [1.0, 0.0],
        dtype=problem_spec.bounds.dtype,
        device=problem_spec.bounds.device,
    )
    objective = ScalarizedObjective(weights=weights)

    aux_acqf = ExpectedImprovement(model=model, best_f=best_f, objective=objective)

    new_x, acq_value = optimize_acqf(
        acq_function=aux_acqf,
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

    posterior = model.posterior(X=new_x)
    mean = posterior.mean
    var = posterior.variance
    mean_cost = torch.exp(mean[..., 1] + 0.5 * var[..., 1])

    multiplier = acq_value / mean_cost
    multiplier = multiplier.item()
    return multiplier


def initialize_model(
    train_x: Tensor,
    train_y: Tensor,
    problem_spec: ProblemSpec,
    training_mode: str,
    noiseless_obs: bool = False,
    log_transform_y: bool = False,
    log_scale_acqf: bool = False,
    multi_fidelity: bool = False,
):
    train_x_normalized = normalize(train_x, bounds=problem_spec.bounds)
    if train_y.ndim == 1:
        train_y = train_y.unsqueeze(-1)

    n_outputs = train_y.shape[-1]

    if training_mode == "objective":
        log_transform_y_adjusted = False
    else:
        log_transform_y_adjusted = log_transform_y

    # outcome transform
    batch_shape = train_y.shape[:-2]
    standardize = Standardize(m=n_outputs, batch_shape=batch_shape)

    if log_transform_y_adjusted:
        if not log_scale_acqf:
            if training_mode == "objective_and_cost":
                log = Log(outputs=[1])
            elif training_mode == "cost":
                log = Log()
            outcome_transform = ChainedOutcomeTransform(
                log=log, standardize=standardize
            )
        else:
            train_y[..., -1] = torch.log(train_y[..., -1])
            outcome_transform = standardize
    else:
        outcome_transform = standardize

    if noiseless_obs:
        _, aug_batch_shape = SingleTaskGP.get_batch_dimensions(
            train_X=train_x_normalized,
            train_Y=train_y,
        )
        likelihood = GaussianLikelihood(
            batch_shape=aug_batch_shape,
            noise_constraint=Interval(lower_bound=1e-4, upper_bound=1e-3),
        )
    else:
        likelihood = None
    # define model
    if multi_fidelity:
        logging.info("Using SingleTaskMultiFidelityGP model")
        model = SingleTaskMultiFidelityGP(
            train_X=train_x_normalized,
            train_Y=train_y,
            data_fidelity=-1,
            linear_truncated=False,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
        )
    else:
        model = SingleTaskGP(
            train_X=train_x_normalized,
            train_Y=train_y,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
        )
    model.outcome_transform.eval()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model
