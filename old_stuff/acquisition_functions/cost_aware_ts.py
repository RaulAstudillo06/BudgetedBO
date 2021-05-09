#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Optional, Union

from botorch.generation import MaxPosteriorSampling
from botorch.models.model import Model
from torch import Tensor

from .objective import RatioMCObjective


class CostAwareThompsonSampling(MaxPosteriorSampling):
    r"""Cost-Aware Thompson Sampling sampling strategy."""

    def __init__(
        self,
        model: Model,
        replacement: bool = True,
        reference_val: Union[float, Tensor] = 0.0,
        min_cost: Optional[Union[float, Tensor]] = None,
    ) -> None:
        r"""Constructor for the CostAwareThmpsonSampling class.

        Args:
            model: A fitted model. Unless a tailored objective is passed, it assumes
                that the model has two outcomes, the first one corresponding to the
                objective function, and the second one corresponding to the cost
                function.
            objective: The objective. Typically, the AcquisitionObjective under which
                the samples are evaluated. If a ScalarizedObjective, samples from the
                scalarized posterior are used. Defaults to the objective that divides
                the first outcome by the maximum between min_cost and the second outcome.
            replacement: If True, sample with replacement.
            min_cost: Scalar representing the minimum allowable value of the cost.
        """
        self.reference_val = reference_val
        self.min_cost = min_cost
        objective = RatioMCObjective(
            positive_part=True,
            reference_val=self.reference_val,
            min_denominator=self.min_cost,
        )
        super().__init__(model=model, objective=objective, replacement=replacement)
