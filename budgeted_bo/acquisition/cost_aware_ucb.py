#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Optional, Union

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch import Tensor


class CostAwareUpperConfidenceBound(AcquisitionFunction):
    r"""Cost-aware (CA) Upper Confidence Bound (UCB) acquisition function.

    This class implements a cost-aware version of the UCB acquisition
    function, obtained by dividing the classical UCB acquisition by
    an analogous lower confidence bound of the cost.

    `CA-UCB(x) = {mu_f(x) + sqrt(beta_f) * sigma_f(x) - reference_val}_+ /
                 (mu_c(x) - sqrt(beta_c) * sigma_c(x))`,
    where `mu_f,` `mu_c`, `sigma_f`, and `sigma_c` are the posterior means and standard
    deviations of the objective and cost functions, respectively.
    """

    def __init__(
        self,
        model: Model,
        beta_obj: Union[float, Tensor],
        beta_cost: Union[float, Tensor],
        maximize: bool = True,
        is_log_cost: bool = True,
        reference_val: Union[float, Tensor] = 0.0,
        min_cost: Optional[Union[float, Tensor]] = None,
    ) -> None:
        r"""Cost-Aware Upper Confidence Bound.
        Args:
            model: Fitted two-output GP model. Assumes that the first output
                corresponds to the objective and the second output to the cost.
            beta_obj: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance for the
                numerator.
            beta_cost: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance for the
                denominator (prevents it from being zero or negative).
            min_cost: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the minimum value of the denominator (prevents it from being
                zero or negative).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model)

        beta_obj = torch.tensor(beta_obj)
        self.register_buffer("beta_obj", beta_obj)

        beta_cost = torch.tensor(beta_cost)
        self.register_buffer("beta_cost", beta_cost)

        self.maximize = maximize
        self.is_log_cost = is_log_cost
        reference_val = torch.tensor(reference_val)
        self.register_buffer("reference_val", reference_val)
        if min_cost is not None:
            min_cost = torch.tensor(min_cost)
            self.register_buffer("min_cost", min_cost)
        else:
            self.min_cost = min_cost

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Cost-Aware Upper Confidence Bound on the candidate set X.
        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
        Returns:
            A `(b)`-dim Tensor of cost-aware upper confidence bound values at the given
            design points `X`.
        """
        posterior = self.model.posterior(X=X)
        batch_shape = X.shape[:-2]

        self.beta_obj = self.beta_obj.to(X)
        mean_obj = posterior.mean[..., 0].view(batch_shape)
        var_obj = posterior.variance[..., 0].view(batch_shape)
        delta_obj = (self.beta_obj.expand_as(mean_obj) * var_obj).sqrt()
        if self.maximize:
            obj_cb = mean_obj + delta_obj
        else:
            obj_cb = mean_obj - delta_obj

        self.beta_cost = self.beta_cost.to(X)
        mean_cost = posterior.mean[..., 1].view(batch_shape)
        var_cost = posterior.variance[..., 1].view(batch_shape)
        delta_cost = (self.beta_cost.expand_as(mean_cost) * var_cost).sqrt()
        cost_cb = mean_cost - delta_cost
        if self.is_log_cost:
            cost_cb = torch.exp(cost_cb)
        if self.min_cost is not None:
            cost_cb = torch.max(self.min_cost, cost_cb)

        return (obj_cb - self.reference_val).clamp_min(0.0) / cost_cb
