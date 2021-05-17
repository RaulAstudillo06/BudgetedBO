#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Optional, Union

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.models.model import Model
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal


class ExpectedImprovementPerUnitOfCost(AnalyticAcquisitionFunction):
    r"""Expected Improvement Per Unit of Cost (analytic).
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        cost_exponent: Union[float, Tensor] = 1.0,
        maximize: bool = True,
        objective: Optional[ScalarizedObjective] = None,
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).
        Args:
            model: A fitted two-outcome model, where the first output corresponds
                to the objective and the second one to the log-cost.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            maximize: If True, consider the problem a maximization problem.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.objective = None
        self.maximize = maximize
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("cost_exponent", torch.as_tensor(cost_exponent))

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement Per Unit of Cost on the candidate set X.
        Args:
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.
        Returns:
            A `b1 x ... bk`-dim tensor of Expected Improvement Per Unit of Cost values 
            at the given design points `X`.
        """
        posterior = self._get_posterior(X=X)
        means = posterior.mean  # (b) x 2
        vars = posterior.variance.clamp_min(1e-6) # (b) x 2
        stds = vars.sqrt()

        # (b) x 1
        mean_obj = means[..., 0]
        std_obj = stds[..., 0]
        u = (mean_obj - self.best_f) / std_obj

        if not self.maximize:
            u = -u
        standard_normal = Normal(
            torch.zeros(1, device=u.device, dtype=u.dtype),
            torch.ones(1, device=u.device, dtype=u.dtype),
        )
        pdf_u = torch.exp(standard_normal.log_prob(u))
        cdf_u = standard_normal.cdf(u)
        ei = std_obj * (pdf_u + u * cdf_u)  # (b) x 1
        # (b) x 1
        eic = torch.exp(-(self.cost_exponent * means[..., 1]) + 0.5 * (torch.square(self.cost_exponent) * vars[..., 1]))
        ei_puc = ei.mul(eic)  # (b) x 1
        return ei_puc.squeeze(dim=-1)