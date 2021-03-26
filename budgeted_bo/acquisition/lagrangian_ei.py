#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Optional, Union

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal


class LagrangianExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Analytic Lagrangian Expected Improvement.

    Computes the analytic expected improvement minus the expected cost times the
    value of the Lagrange multiplier. The objective and (log-) cost are assumed to
    have Gaussian posterior distributions. Only supports the case `q=1`. The model
    should be two-outcome, with the first output corresponding to the objective and
    the second output corresponding to the log-cost.
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        multiplier: Union[float, Tensor],
        maximize: bool = True,
        objective: Optional[ScalarizedObjective] = None,
    ) -> None:
        r"""Analytic Lagrangian Expected Improvement.
        Args:
            model: A fitted two-outcome model, where the first output corresponds
                to the objective and the second one to the log-cost.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            multiplier: A scalar representing the Lagrange multiplier.
            maximize: If True, consider the problem a maximization problem.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.objective = None
        self.maximize = maximize
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("multiplier", torch.as_tensor(multiplier))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Lagrangian Expected Improvement on the candidate set X.
        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        posterior = self._get_posterior(X=X)
        means = posterior.mean  # (b) x 2
        sigmas = posterior.variance.sqrt().clamp_min(1e-6)  # (b) x 2

        # (b) x 1
        mean_obj = means[..., 0]
        sigma_obj = sigmas[..., 0]
        u = (mean_obj - self.best_f) / sigma_obj

        if not self.maximize:
            u = -u
        standard_normal = Normal(
            torch.zeros(1, device=u.device, dtype=u.dtype),
            torch.ones(1, device=u.device, dtype=u.dtype),
        )
        pdf_u = torch.exp(standard_normal.log_prob(u))
        cdf_u = standard_normal.cdf(u)
        ei = sigma_obj * (pdf_u + u * cdf_u)  # (b) x 1
        # (b) x 1
        mean_cost = torch.exp(means[..., 1] + 0.5 * sigmas[..., 1])

        penalized_ei = ei - self.multiplier * mean_cost  # (b) x 1
        return penalized_ei.squeeze(dim=-1)
