#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from torch import Tensor


class NaiveCostAwareAcquisitionFunction(AcquisitionFunction):
    r"""Naive cost-aware acquisition function.
    This class implements a naive cost-aware acquisition function, where a raw
    acquisition function is divided by the expected cost. This technique makes
    sense when the raw acquisition function is motivated by some notion of
    "improvement" (e.g., EI, PI), so that the cost-aware version represents
    "improvement" per "cost".

    .. [Swersky2013multitask]
        K. Swersky, J. Snoek, and R.P. Adams. Multi-task Bayesian optimization. NeurIPS 2013.
    """

    def __init__(
        self,
        raw_acqf: AcquisitionFunction,
        cost_model: Model,
        use_mean: bool = True,
        cost_exponent: float = 1.0,
        min_cost: Optional[float] = None,
        cost_sampler: Optional[MCSampler] = None,
        cost_objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        r""".
        Args:
            raw_acqf: Raw acquisition function to be divided by the expected
                cost.
            cost_model: Fitted single-outcome GP model for (a transformation of)
                the cost function.
            is_log_cost: Flag that indicates if the log cost is being modeled;
                used to compute the expected cost analytically.
            cost_sampler: A sampler used for sampling from the posterior of the
                cost model.
            cost_objective: If specified, transform the posterior samples from the
                cost model. This can be used e.g. to un-transform predictions/samples
                of a cost model fit on the log-transformed cost (often done to ensure
                non-negativity).
        """
        super().__init__(model=raw_acqf.model)
        self.raw_acqf = raw_acqf
        self.cost_model = cost_model
        self._use_mean = use_mean
        self._cost_exponent = cost_exponent
        self._min_cost = min_cost
        if cost_sampler is None and not use_mean:
            cost_sampler = SobolQMCNormalSampler(
                num_samples=512, collapse_batch_dims=True
            )
        self.cost_sampler = cost_sampler
        if cost_objective is None:
            cost_objective = IdentityMCObjective()
        self.cost_objective = cost_objective

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set `X`.
        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each. Note that some raw acquisition functions may not support
                `q > 1` (e.g., analytic acquisition functions).
        Returns:
            A `batch_shape'`-dim Tensor of Acquisition values at the given design points
            `X`, where `batch_shape'` is the broadcasted batch shape of model and input `X`.
            These values are obtained by dividing the raw acquisition by the expected cost.
        """
        raw_value = self.raw_acqf(X=X)
        cost_posterior = self.cost_model.posterior(X)
        if self._use_mean:
            expected_cost = cost_posterior.mean.squeeze(dim=-1)
            if self._min_cost is not None:
                expected_cost = expected_cost.clamp_min(self._min_cost)
            # add cost of points in q-batch
            expected_cost = expected_cost.sum(dim=-1)
        else:
            cost_samples = self.cost_sampler(cost_posterior)
            # transform cost samples
            cost_samples = self.cost_objective(cost_samples)
            if self._min_cost is not None:
               cost_samples = cost_samples.clamp_min(self._min_cost)
            # add cost of points in q-batch
            cost_samples = cost_samples.sum(dim=-1)
            expected_cost = cost_samples.mean(dim=0)
        # divide raw acquisition value by expected cost
        value = raw_value / torch.pow(expected_cost, self._cost_exponent)
        return value

    @property
    def X_pending(self) -> Optional[Tensor]:
        r"""Informs the acquisition function about pending design points.
        Returns:
            X_pending: `n x d` Tensor with `n` `d`-dim design points that have been submitted
            for evaluation but have not yet been evaluated. Note that pending points are
            handled through the raw acquisition function only. Some raw acquisition functions
            may not support pending points (e.g., analytic acquisition functions).
        """
        return self.raw_acqf.X_pending

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Informs the acquisition function about pending design points.
        Args:
            X_pending: `n x d` Tensor with `n` `d`-dim design points that have been submitted
            for evaluation but have not yet been evaluated. Note that pending points are
            handled through the raw acquisition function only. Some raw acquisition functions
            may not support pending points (e.g., analytic acquisition functions).
        """
        self.raw_acqf.set_X_pending(X_pending=X_pending)
