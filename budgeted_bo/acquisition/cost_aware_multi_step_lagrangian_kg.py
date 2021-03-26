#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Callable, List, Optional, Type, Union

from botorch.acquisition.objective import AcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch_fb.experimental.acquisition.cost_aware_multi_step import (
    qCostAwareMultiStepLookahead,
)
from botorch_fb.experimental.acquisition.cost_aware_utility import (
    CumulativeCostLagrangianUtility,
)
from torch import Tensor


class qCostAwareMultiStepLookaheadLagrangianKG(qCostAwareMultiStepLookahead):
    r"""MC-based batch Cost-Aware Multi-Step Ratio Knowledge Gradient."""

    def __init__(
        self,
        model: Model,
        multiplier: Union[float, Tensor],
        budget: Union[float, Tensor],
        batch_sizes: List[int],
        num_fantasies: Optional[List[int]] = None,
        project: Optional[Type[Callable]] = None,
        samplers: Optional[List[MCSampler]] = None,
        objective: Optional[AcquisitionObjective] = None,
        inner_mc_samples: Optional[List[int]] = None,
        X_pending: Optional[Tensor] = None,
        collapse_fantasy_base_samples: bool = True,
    ) -> None:
        r"""Cost-Aware Multi-Step Look-Ahead Ratio Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted two-output model, where the first output corresponds to the
                objective, and the second one to the cost.
            batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes for the
                `k` look-ahead steps.
            multiplier: Value of multiplier in Lagrangian term.
            budget: Budget constraint used in Lagrangian term.
            num_fantasies: A list `[f_1, ..., f_k]` containing the number of fantasy
                points to use for the `k` look-ahead steps.
            num_fantasies: A list `[f_1, ..., f_k]` containing the number of fantasy
                points to use for the `k` look-ahead steps.
            project: Callable that maps `X_k` to a tensor of the same shape projected to
                the desired target set (e.g. target fidelities in case of multi-fidelity
                optimization).
            samplers: A list of MCSampler objects to be used for sampling fantasies in
                each stage.
            objective: The objective under which the output is evaluated. If `None`, use
                the model output (requires a single-output model). If a
                `ScalarizedObjective` and `value_function_cls` is a subclass of
                `AnalyticAcquisitonFunction`, then the analytic posterior mean is used.
                Otherwise the objective is MC-evaluated (using `inner_sampler`).
            inner_mc_samples: A list `[n_0, ..., n_k]` containing the number of MC
                samples to be used for evaluating the stage value function. Ignored if
                the objective is `None` or a `ScalarizedObjective`.
            X_pending: A `m x d`-dim Tensor of `m` design points that have points that
                have been submitted for function evaluation but have not yet been
                evaluated. Concatenated into `X` upon forward call. Copied and set to
                have no gradient.
            collapse_fantasy_base_samples: If True, collapse_batch_dims of the Samplers
                will be applied on fantasy batch dimensions as well, meaning that base
                samples are the same in all subtrees starting from the same level.
        """
        self.multiplier = multiplier
        self.budget = budget

        cost_aware_utilities = [None] * len(batch_sizes)
        cumulative_cost_lagrangian_utility = CumulativeCostLagrangianUtility(
            multiplier=self.multiplier, budget=self.budget, batch_sizes=batch_sizes
        )
        cost_aware_utilities.append(cumulative_cost_lagrangian_utility)

        super().__init__(
            model=model,
            batch_sizes=batch_sizes,
            num_fantasies=num_fantasies,
            samplers=samplers,
            cost_aware_utilities=cost_aware_utilities,
            project=project,
            objective=objective,
            inner_mc_samples=inner_mc_samples,
            X_pending=X_pending,
            collapse_fantasy_base_samples=collapse_fantasy_base_samples,
        )
