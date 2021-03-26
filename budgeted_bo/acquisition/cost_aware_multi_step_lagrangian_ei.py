#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import List, Optional, Union

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.objective import AcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch_fb.experimental.acquisition.cost_aware_multi_step import (
    make_best_f_ca,
    qCostAwareMultiStepLookahead,
)
from botorch_fb.experimental.acquisition.cost_aware_utility import (
    ExpectedCostLagrangianUtility,
)
from torch import Tensor


class CostAwareMultiStepLookaheadLagrangianEI(qCostAwareMultiStepLookahead):
    r"""MC-based batch Cost-Aware Multi-Step Lagrangian Expected Improvement (q=1)."""

    def __init__(
        self,
        model: Model,
        multiplier: Union[float, Tensor],
        budget: Union[float, Tensor],
        num_fantasies: Optional[List[int]] = None,
        samplers: Optional[List[MCSampler]] = None,
        objective: Optional[AcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
        collapse_fantasy_base_samples: bool = True,
    ) -> None:
        r"""Cost-Aware Multi-Step Look-Ahead Ratio Expected Improvement (q=1).

        Args:
            model: A fitted two-output model, where the first output corresponds to the
                objective, and the second one to the cost.
            multiplier: Value of multiplier in Lagrangian term.
            budget: Budget constraint used in Lagrangian term.
            num_fantasies: A list `[f_1, ..., f_k]` containing the number of fantasy
                points to use for the `k` look-ahead steps.
            samplers: A list of MCSampler objects to be used for sampling fantasies in
                each stage.
            objective: The objective under which the output is evaluated. If `None`, use
                the model output (requires a single-output model). If a
                `ScalarizedObjective` and `value_function_cls` is a subclass of
                `AnalyticAcquisitonFunction`, then the analytic posterior mean is used.
                Otherwise the objective is MC-evaluated (using `inner_sampler`).
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

        n_lookahead_steps = len(num_fantasies)

        batch_sizes = [1 for _ in range(n_lookahead_steps)]
        valfunc_cls = [ExpectedImprovement for _ in range(n_lookahead_steps + 1)]
        valfunc_argfacs = [make_best_f_ca for _ in range(n_lookahead_steps + 1)]
        cost_aware_utilities = [
            ExpectedCostLagrangianUtility(
                multiplier=self.multiplier, budget=self.budget / (n_lookahead_steps + 1)
            )
            for _ in range(n_lookahead_steps + 1)
        ]

        super().__init__(
            model=model,
            batch_sizes=batch_sizes,
            num_fantasies=num_fantasies,
            samplers=samplers,
            valfunc_cls=valfunc_cls,
            valfunc_argfacs=valfunc_argfacs,
            cost_aware_utilities=cost_aware_utilities,
            objective=objective,
            X_pending=X_pending,
            collapse_fantasy_base_samples=collapse_fantasy_base_samples,
        )
