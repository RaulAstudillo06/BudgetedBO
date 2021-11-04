#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Optional, Union

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils.objective import soft_eval_constraint
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal


class BudgetedExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Analytic Budgeted Expected Improvement.

    Computes the analytic expected improvement weighted by a probability of
    satisfying a budget constraint. The objective and (log-) cost are assumed
    to be independent and have Gaussian posterior distributions. Only supports
    the case `q=1`. The model should be two-outcome, with the first output
    corresponding to the objective and the second output corresponding to the
    log-cost.
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        budget: Union[float, Tensor],
        maximize: bool = True,
        beta: Optional[Union[float, Tensor]] = None,
        objective: Optional[ScalarizedObjective] = None,
    ) -> None:
        r"""Analytic Budgeted Expected Improvement.
        Args:
            model: A fitted two-outcome model, where the first output corresponds
                to the objective and the second one to the log-cost.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            budget: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the budget constraint.
            maximize: If True, consider the problem a maximization problem.
            beta: A scalar representing the hyperparameter of the soft-plus
                transformation applied to the budget. Default is None.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.objective = None
        self.maximize = maximize
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("budget", torch.as_tensor(budget))
        if beta is not None:
            self.register_buffer("beta", torch.as_tensor(beta))
        else:
            self.beta = beta

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.
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
        prob_feas = self._compute_prob_feas(means=means[..., 1], sigmas=sigmas[..., 1])
        bc_ei = ei.mul(prob_feas)  # (b) x 1
        return bc_ei.squeeze(dim=-1)

    def _compute_prob_feas(self, means: Tensor, sigmas: Tensor) -> Tensor:
        r"""Compute feasibility probability for each batch of X.
        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
            means: A `(b) x 1`-dim Tensor of means.
            sigmas: A `(b) x 1`-dim Tensor of standard deviations.
        Returns:
            A `(b) x 1`-dim tensor of feasibility probabilities.
        """
        standard_normal = Normal(
            torch.zeros(1, device=means.device, dtype=means.dtype),
            torch.ones(1, device=means.device, dtype=means.dtype),
            validate_args=True,
        )
        if self.beta is not None:
            soft_plus = torch.nn.Softplus(beta=self.beta)
            soft_plus_budget = soft_plus(self.budget)
            prob_feas = standard_normal.cdf(
                (torch.log(soft_plus_budget) - means) / sigmas
            )
        else:
            prob_feas = standard_normal.cdf(
                (torch.log(self.budget.clamp_min(1e-6)) - means) / sigmas
            )
            prob_feas = torch.where(
                self.budget > 1e-6,
                prob_feas,
                torch.zeros(1, device=means.device, dtype=means.dtype),
            )
        return prob_feas


class qBudgetedExpectedImprovement(MCAcquisitionFunction):
    r"""Batch Budget-Constrained Expected Improvement.

    Computes the expected improvement weighted by a probability of satisfying
    a budget constraint. The model should be two-outcome, with the first output
    corresponding to the objective and the second output corresponding to the
    log-cost.
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        budget: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""Batch Budgeted Expected Improvement.
        Args:
            model: A fitted two-outcome model, where the first output corresponds
                to the objective and the second one to the log-cost.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            log_budget: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the budget constraint.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
        """
        # use AcquisitionFunction constructor to avoid check for objective
        super(MCAcquisitionFunction, self).__init__(model=model)
        self.sampler = sampler
        self.objective = None
        self.X_pending = X_pending
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("budget", torch.as_tensor(budget))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        improvements = (samples[..., 0] - self.best_f).clamp_min(0)
        max_improvement = improvements.max(dim=-1, keepdim=True)[0]
        sum_costs = torch.exp(samples[..., 1]).sum(dim=-1, keepdim=True)
        smooth_feas_ind = soft_eval_constraint(lhs=sum_costs - self.budget)
        bc_ei = torch.mul(max_improvement, smooth_feas_ind).mean(dim=0)
        return bc_ei.squeeze(dim=-1)
