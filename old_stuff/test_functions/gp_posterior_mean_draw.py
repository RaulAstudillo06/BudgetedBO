#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Callable

import gpytorch
import torch
from botorch.models import FixedNoiseGP
from botorch.utils.sampling import manual_seed
from gpytorch.kernels.matern_kernel import MaternKernel


def gp_posterior_mean_draw(
    input_dim: int,
    output_dim: int,
    n_fantasies: int,
    seed: int,
) -> Callable:
    r"""Returns an approximate sample from a GP model. # TODO: Allow passing as input a GP model

    This function returns an approximate sample from GP model. This approximate
    sample is given by the posterior mean of a fantasy model obtained by drawing
    fantasy samples over a set of points chosen uniformly at random.

    Args:
        input_dim: Input dimension of the GP model.
        output_dim: Number of outputs of the GP model.
        n_fantasies: Number of samples used to build fantasy model.
        seed: An integer used to fix the stream of random numbers used to generate
            the GP sample.
    """

    with manual_seed(seed=seed):
        X_fantasy = torch.rand((n_fantasies, input_dim))
        Y_aux = torch.empty((n_fantasies, output_dim))
        covar_module = MaternKernel(
            nu=2.5,
            batch_shape=torch.Size([2]),
            ard_num_dims=X_fantasy.shape[-1],
        )
        covar_module._set_lengthscale(value=0.1)
        model = FixedNoiseGP(
            X_fantasy,
            Y_aux,
            torch.full_like(Y_aux, 1e-5),
            covar_module=covar_module,
        )
        with gpytorch.settings.prior_mode(True):
            prior_at_X_aux = model.posterior(X_fantasy)

        Y_from_prior = prior_at_X_aux.rsample(torch.Size([])).detach()
        fantasy_model = FixedNoiseGP(
            X_fantasy,
            Y_from_prior,
            torch.full_like(Y_from_prior, 1e-5),
            covar_module=covar_module,
        )

    def fantasy_posterior_mean(X):
        posterior_at_X = fantasy_model.posterior(X)
        posterior_mean_at_X = posterior_at_X.mean.detach()
        return posterior_mean_at_X

    return fantasy_posterior_mean
