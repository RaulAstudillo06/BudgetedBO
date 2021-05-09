#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from copy import copy
from typing import List, Optional, Union

import torch
from botorch.acquisition.cost_aware import CostAwareUtility
from botorch.models.model import Model
from torch import Tensor


# TODO: Add support for cost objectives


class ValCostRatioUtility(CostAwareUtility):
    r"""A cost-aware utility that divides the deltas (optionally, deltas minus a reference value) by
    the observed cumulative cost.

    `U(deltas) = (deltas - reference_value) / cumulative_cost`
    """

    def __init__(
        self,
        batch_sizes: List[int],
        batch_max_cost: bool = False,
        reference_value: Union[float, Tensor] = 0.0,
        min_cost: Optional[Union[float, Tensor]] = None,
    ) -> None:
        r"""Cost-aware utility that divides the deltas (optionally, deltas minus a reference value) by
        the observed cumulative cost.
        Args:
            batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes of the observed costs.
            batch_max_cost: If True, only the maximum cost in each batch is added to the cumulative
                cost; otherwise, the sum of the costs in each batch is added.
            reference_value: A value to be substracted from the deltas.
            min_cost: A value used to clamp the observed costs.
        """
        super().__init__()
        self.batch_sizes = batch_sizes
        self.batch_max_cost = batch_max_cost
        self.reference_value = reference_value
        if min_cost is not None:
            min_cost = torch.tensor(min_cost)
            self.register_buffer("min_cost", min_cost)
        else:
            self.min_cost = min_cost
        self.n_evals = sum(batch_sizes)

    def forward(self, X: Tensor, deltas: Tensor, model: Model) -> Tensor:
        r"""Evaluate utility function on the given deltas.
        Args:
            X: A `batch_shape x q x d`-dim Tensor of with `q` `d`-dim design
                points each for each t-batch.
            deltas: A `batch_shape`-dim Tensor of samples from the reward over
                the current state at `X` for each t-batch.
            model: A batched Model of batch size `batch_shape`. It must be possible
                to evaluate the model's posterior at `X`.
        Returns:
            A `batch_shape`-dim Tensor of transformed deltas.
        """
        y = torch.transpose(model.train_targets, -2, -1)
        y_original_scale = model.outcome_transform.untransform(y)[0]
        log_costs = y_original_scale[..., 1]
        costs = torch.exp(log_costs)
        if self.min_cost is not None:
            costs.clamp_min(self.min_cost)
        cumulative_costs = _compute_cumulative_cost(
            costs, self.batch_sizes, self.batch_max_cost
        )
        return (deltas - self.reference_value) / cumulative_costs


class ExpectedCostLagrangianUtility(CostAwareUtility):
    r"""A cost-aware utility that substracts from deltas a Lagrangian term
    corresponding to a constraint of the form `mean_cost(X) <= budget`.

    `U(deltas) = deltas - mean_{cost}(X)`
    """

    def __init__(
        self, multiplier: Union[float, Tensor], budget: Union[float, Tensor]
    ) -> None:
        r"""Cost-aware utility that substracts from deltas a Lagrangian term
        corresponding to a the constraint of the form `mean_cost(X) <= budget`.
        Args:
            budget: A value indicating a constraint on the cumulative cost. This constraint gives
                raise to the Lagrangian term in the utility function.
            batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes of the observed costs.
            batch_max_cost: If True, only the maximum cost in each batch is added to the cumulative
                cost; otherwise, the sum of the costs in each batch is added.
        """
        super().__init__()
        multiplier = torch.tensor(multiplier)
        self.register_buffer("multiplier", multiplier)
        budget = torch.tensor(budget)
        self.register_buffer("budget", budget)

    def forward(self, X: Tensor, deltas: Tensor, model=Model) -> Tensor:
        r"""Evaluate utility function on the given deltas.
        Args:
            X: A `num_fantasies x batch_shape x q x d`-dim Tensor of with `q` `d`-dim design
                points each for each t-batch.
            deltas: A `num_fantasies x batch_shape`-dim Tensor of samples from the reward over
                the current state at `X` for each t-batch.
            model: model: A batched Model of batch size `num_fantasies`. It must be possible
                to evaluate the model's posterior at `X`.
        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of transformed deltas.
        """
        posterior = model.posterior(X)
        cost_mean = posterior.mean[..., 1].squeeze(dim=-1)
        return deltas - self.multiplier * (cost_mean - self.budget)


class CumulativeCostLagrangianUtility(CostAwareUtility):
    r"""Cost-aware utility that substracts from deltas a Lagrangian term corresponding to a budget
    constraint on the cumulative cost.

    `U(deltas) = deltas - multiplier * (cumulative_cost - budget)`
    """

    def __init__(
        self,
        multiplier: Union[float, Tensor],
        budget: Union[float, Tensor],
        batch_sizes: List[int],
        batch_max_cost: bool = False,
    ) -> None:
        r"""Cost-aware utility that substracts from deltas a Lagrangian term corresponding to a budget
        constraint on the cumulative cost.
        Args:
            multiplier: Value of the Lagrange multiplier.
            budget: A value indicating a budget constraint on the cumulative cost. This constraint gives
                raise to the Lagrangian term in the utility function.
            batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes of the observed costs.
            batch_max_cost: If True, only the maximum cost in each batch is added to the cumulative
                cost; otherwise, the sum of the costs in each batch is added.
        """
        super().__init__()
        multiplier = torch.tensor(multiplier)
        self.register_buffer("multiplier", multiplier)
        budget = torch.tensor(budget)
        self.register_buffer("budget", budget)
        self.batch_sizes = batch_sizes
        self.batch_max_cost = batch_max_cost
        self.n_evals = sum(batch_sizes)

    def forward(self, X: Tensor, deltas: Tensor, model: Model) -> Tensor:
        r"""Evaluate utility function on the given deltas.
        Args:
            X: A `num_fantasies x batch_shape x q x d`-dim Tensor of with `q` `d`-dim design
                points each for each t-batch.
            deltas: A `num_fantasies x batch_shape`-dim Tensor of samples from the reward over
                the current state at `X` for each t-batch.
            model: model: A batched Model of batch size `num_fantasies`. It must be possible
                to evaluate the model's posterior at `X`.
        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of transformed deltas.
        """
        costs = model.train_targets[1][..., -self.n_evals - 1 : -1]
        cumulative_costs = _compute_cumulative_cost(
            costs, self.batch_sizes, self.batch_max_cost
        )
        return deltas - self.multiplier * (cumulative_costs - self.budget)


def _compute_cumulative_cost(
    costs: Tensor, batch_sizes: List, batch_max_cost: bool = False
):
    r"""Computes the cumulative cost.
    Args:
        costs: A `batch_shape x n_observations` Tensor of observed costs, where
            `n_observations = sum(batch_sizes)`.
        batch_sizes: A list `[q_1, ..., q_k]` containing the q-batch sizes of the observed costs.
        batch_max_cost: If True, only the maximum cost in each batch is added to the cumulative
            cost; otherwise, the sum of the costs in each batch is added.
    Returns:
        A `batch_shape` Tensor of cumulative costs.
    """
    if batch_max_cost:
        batch_start = 0
        batch_end = batch_sizes[0]
        cumulative_cost = torch.max(costs[..., batch_start:batch_end], dim=-1)
        for q in batch_sizes[1:]:
            batch_start = copy(batch_end)
            batch_end += q
            cumulative_cost += torch.max(costs[..., batch_start:batch_end], dim=-1)
    else:
        cumulative_cost = torch.sum(costs, dim=-1)
    return cumulative_cost
