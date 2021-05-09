#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.cost_aware import CostAwareUtility
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_step_lookahead import (
    TAcqfArgConstructor,
    _compute_stage_value,
    _construct_sample_weights,
    qMultiStepLookahead,
)
from botorch.acquisition.objective import (
    AcquisitionObjective,
    LinearMCObjective,
    ScalarizedObjective,
)
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform
from torch import Tensor


class qCostAwareMultiStepLookahead(qMultiStepLookahead):
    r"""MC-based batch Cost-Aware Multi-Step Look-Ahead (one-shot optimization)."""

    def __init__(
        self,
        model: Model,
        batch_sizes: List[int],
        num_fantasies: Optional[List[int]] = None,
        samplers: Optional[List[MCSampler]] = None,
        valfunc_cls: Optional[List[Optional[Type[AcquisitionFunction]]]] = None,
        valfunc_argfacs: Optional[List[Optional[TAcqfArgConstructor]]] = None,
        cost_aware_utilities: Optional[List[Optional[Type[CostAwareUtility]]]] = None,
        terminal_cost_aware_utility: Optional[Type[CostAwareUtility]] = None,
        project: Optional[Type[Callable]] = None,
        objective: Optional[AcquisitionObjective] = None,
        inner_mc_samples: Optional[List[int]] = None,
        X_pending: Optional[Tensor] = None,
        collapse_fantasy_base_samples: bool = True,
    ) -> None:
        r"""Cost-Aware Multi-Step Look-Ahead Ratio Knowledge Gradient (one-shot optimization).

        Performs a `k`-step lookahead by means of repeated fantasizing.

        Allows to specify a list of cost-aware utilities so that, at each stage, the stage
        value is transformed by its corresponding cost-aware utility to induce
        cost-awareness.

        WARNING: The complexity of evaluating this function is exponential in the number
        of lookahead steps!

        Args:
            model: A fitted multi-output model, where the first output corresponds to the
                objective, and the remaining ones to the (potentially various sources of) costs.
            batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes for the
                `k` look-ahead steps.
            num_fantasies: A list `[f_1, ..., f_k]` containing the number of fantasy
                points to use for the `k` look-ahead steps.
            samplers: A list of MCSampler objects to be used for sampling fantasies in
                each stage.
            valfunc_cls: A list of `k + 1` acquisition function classes to be used as
                the (stage + terminal) value functions. Each element (except for the
                last one) can be `None`, in which case a zero stage value is assumed for
                the respective stage. If `None`, this defaults to
                `[None, ..., None, PosteriorMean]`
            valfunc_argfacs: A list of `k + 1` "argument factories", i.e. callables that
                map a `Model` and input tensor `X` to a dictionary of kwargs for the
                respective stage value function constructor (e.g. `best_f` for
                `ExpectedImprovement`). If None, only the standard (`model`, `sampler`
                and `objective`) kwargs will be used.
            cost_aware_utilities: A list of 'k + 1` cost-aware utilities that transform stage
                    and terminal values to induce cost-awareness.
            terminal_cost_aware_utility: A cost-aware utility that transforms the final
                    running value to induce cost-awareness.
            project: Callable that maps `X_k` to a tensor of the same shape projected to
                the desired target set (e.g. target fidelities in case of multi-fidelity
                optimization).
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

        if objective is None:
            # TODO: Find a proper way to set the type
            weights = torch.zeros(model.num_outputs, dtype=torch.double)
            weights[0] = 1.0
            if valfunc_cls is None or issubclass(
                valfunc_cls[-1], AnalyticAcquisitionFunction
            ):
                objective = ScalarizedObjective(weights=weights)
            elif issubclass(valfunc_cls[-1], MCAcquisitionFunction):
                objective = LinearMCObjective(weights=weights)

        if project is None:

            def project(X):
                return X

        super().__init__(
            model=model,
            batch_sizes=batch_sizes,
            num_fantasies=num_fantasies,
            samplers=samplers,
            valfunc_cls=valfunc_cls,
            valfunc_argfacs=valfunc_argfacs,
            objective=objective,
            inner_mc_samples=inner_mc_samples,
            X_pending=X_pending,
            collapse_fantasy_base_samples=collapse_fantasy_base_samples,
        )
        self._cost_aware_utilities = cost_aware_utilities
        self._terminal_cost_aware_utility = terminal_cost_aware_utility
        self._project = project

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qCostAwareMultiStepLookahead on the candidate set X.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i` is respective number of fantasies.

        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        batch_shape, shapes, sizes = self.get_split_shapes(X=X)
        # # Each X_i in Xsplit has shape batch_shape x qtilde x d with
        # # qtilde = f_i * ... * f_1 * q_i
        Xsplit = torch.split(X, sizes, dim=-2)
        # now reshape (need to permute batch_shape and qtilde dimensions for i > 0)
        perm = [-2] + list(range(len(batch_shape))) + [-1]
        X0 = Xsplit[0].reshape(shapes[0])
        Xother = [
            X.permute(*perm).reshape(shape) for X, shape in zip(Xsplit[1:], shapes[1:])
        ]
        # concatenate in pending points
        if self.X_pending is not None:
            X0 = torch.cat([X0, match_batch_shape(self.X_pending, X0)], dim=-2)

        # set batch_range on samplers if not collapsing on fantasy dims
        if not self._collapse_fantasy_base_samples:
            tbatch_dim_start = -2 - len(batch_shape)
            for s in self.samplers:
                s.batch_range = (tbatch_dim_start, -2)

        return _step(
            model=self.model,
            Xs=[X0] + Xother,
            samplers=self.samplers,
            valfunc_cls=self._valfunc_cls,
            valfunc_argfacs=self._valfunc_argfacs,
            cost_aware_utilities=self._cost_aware_utilities,
            terminal_cost_aware_utility=self._terminal_cost_aware_utility,
            project=self._project,
            inner_samplers=self.inner_samplers,
            objective=self.objective,
            running_val=None,
        )


def _step(
    model: Model,
    Xs: List[Tensor],
    samplers: List[Optional[MCSampler]],
    valfunc_cls: List[Optional[Type[AcquisitionFunction]]],
    valfunc_argfacs: List[Optional[TAcqfArgConstructor]],
    cost_aware_utilities: List[Optional[Type[CostAwareUtility]]],
    terminal_cost_aware_utility: Optional[Type[CostAwareUtility]],
    project: Optional[Type[Callable]],
    inner_samplers: List[Optional[MCSampler]],
    objective: AcquisitionObjective,
    running_val: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
    step_index: int = 0,
) -> Tensor:
    r"""Recursive multi-step look-ahead computation.

    Helper function computing the "value-to-go" of a multi-step lookahead scheme.

    Args:
        model: A Model of appropriate batch size. Specifically, it must be possible to
            evaluate the model's posterior at `Xs[0]`.
        Xs: A list `[X_j, ..., X_k]` of tensors, where `X_i` has shape
            `f_i x .... x f_1 x batch_shape x q_i x d`.
        samplers: A list of `k - j` samplers, such that the number of samples of sampler
            `i` is `f_i`. The last element of this list is considered the
            "inner sampler", which is used for evaluating the objective in case it is an
            MCAcquisitionObjective.
        valfunc_cls: A list of acquisition function class to be used as the (stage +
            terminal) value functions. Each element (except for the last one) can be
            `None`, in which case a zero stage value is assumed for the respective
            stage.
        valfunc_argfacs: A list of callables that map a `Model` and input tensor `X` to
            a dictionary of kwargs for the respective stage value function constructor.
            If `None`, only the standard `model`, `sampler` and `objective` kwargs will
            be used.
        cost_aware_utilities: A list of cost-aware utilities that transform stage and
            terminal values to induce cost-awareness.
        terminal_cost_aware_utility: A cost-aware utility that transforms the final
                    running value to induce cost-awareness.
        project: Callable that maps `X_k` to a tensor of the same shape projected to the
                desired target set (e.g. target fidelities in case of multi-fidelity
                optimization).
        inner_samplers: A list of `MCSampler` objects, each to be used in the stage
            value function at the corresponding index.
        objective: The AcquisitionObjective under which the model output is evaluated.
        running_val: As `batch_shape`-dim tensor containing the current running value.
        sample_weights: A tensor of shape `f_i x .... x f_1 x batch_shape` when called
            in the `i`-th step by which to weight the stage value samples. Used in
            conjunction with Gauss-Hermite integration or importance sampling. Assumed
            to be `None` in the initial step (when `step_index=0`).
        step_index: The index of the look-ahead step. `step_index=0` indicates the
            initial step.

    Returns:
        A `b`-dim tensor containing the multi-step value of the design `X`.
    """
    if len(Xs) != len(samplers) + 1:
        raise ValueError("Must have as many samplers as look-ahead steps")

    if len(Xs) == 1:
        X = project(Xs[0])
    else:
        X = Xs[0]

    if sample_weights is None:  # only happens in the initial step
        sample_weights = torch.ones(*X.shape[:-2], device=X.device, dtype=X.dtype)

    # compute stage value
    stage_val = _compute_stage_value(
        model=model,
        valfunc_cls=valfunc_cls[0],
        X=X,
        objective=objective,
        inner_sampler=inner_samplers[0],
        arg_fac=valfunc_argfacs[0],
    )
    if stage_val is not None:  # update running value
        # if not None, running_val has shape f_{i-1} x ... x f_1 x batch_shape
        # stage_val has shape f_i x ... x f_1 x batch_shape

        # transform stage_val using cost-aware utility
        if cost_aware_utilities[0] is not None:
            stage_val = cost_aware_utilities[0](X=X, deltas=stage_val, model=model)

        # this sum will add a dimension to running_val so that
        # updated running_val has shape f_i x ... x f_1 x batch_shape
        running_val = stage_val if running_val is None else running_val + stage_val

    # base case: no more fantasizing, return value
    if len(Xs) == 1:
        if terminal_cost_aware_utility is not None:
            running_val = terminal_cost_aware_utility(
                X=X, deltas=running_val, model=model
            )
        # compute weighted average over all leaf nodes of the tree
        batch_shape = running_val.shape[step_index:]
        # expand sample weights to make sure it is the same shape as running_val,
        # because we need to take a sum over sample weights for computing the
        # weighted average
        sample_weights = sample_weights.expand(running_val.shape)
        tree_value = (running_val * sample_weights).view(-1, *batch_shape).sum(dim=0)
        return tree_value / sample_weights.view(-1, *batch_shape).sum(dim=0)

    # construct fantasy model (with batch shape f_{j+1} x ... x f_1 x batch_shape)
    prop_grads = step_index > 0  # need to propagate gradients for steps > 0
    fantasy_model = model.fantasize(
        X=X, sampler=samplers[0], observation_noise=True, propagate_grads=prop_grads
    )

    # augment sample weights appropriately
    sample_weights = _construct_sample_weights(
        prev_weights=sample_weights, sampler=samplers[0]
    )

    return _step(
        model=fantasy_model,
        Xs=Xs[1:],
        samplers=samplers[1:],
        valfunc_cls=valfunc_cls[1:],
        valfunc_argfacs=valfunc_argfacs[1:],
        cost_aware_utilities=cost_aware_utilities[1:],
        terminal_cost_aware_utility=terminal_cost_aware_utility,
        project=project,
        inner_samplers=inner_samplers[1:],
        objective=objective,
        sample_weights=sample_weights,
        running_val=running_val,
        step_index=step_index + 1,
    )


def make_best_f_ca(model: Model, X: Tensor) -> Dict[str, Any]:
    r"""Extract the best observed training input from the model."""
    return {"best_f": model.train_targets[0].max(dim=-1).values}
