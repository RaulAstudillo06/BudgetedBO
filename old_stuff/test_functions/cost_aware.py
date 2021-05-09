#! /usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

r"""
Cost-aware optimization benchmark problems.

References
.. [Surjanovic2013]
    https://www.sfu.ca/~ssurjano/optimization.html

.. [Wu2019]
    Wu, J., Toscano-Palmerin, S., Frazier, P. I., & Wilson, A. G. (2020, August).
    Practical multi-fidelity bayesian optimization for hyperparameter tuning.
    In Uncertainty in Artificial Intelligence (pp. 788-798). PMLR.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import Callable, Optional

import torch
from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from botorch.test_functions.synthetic import Ackley, Bukin, DropWave, Rosenbrock
from botorch.utils.sampling import manual_seed
from botorch_fb.experimental.test_functions.bandits_ucb_simulator import (
    bandits_ucb_simulator,
)
from botorch_fb.experimental.test_functions.gp_posterior_mean_draw import (
    gp_posterior_mean_draw,
)
from torch import Tensor


class CostAwareTestProblem(BaseTestProblem):
    r"""Base class for cost-aware test problems."""

    def __init__(
        self,
        cost_function: Optional[Callable] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""Base constructor for cost-aware test problems.

        # TODO: Allow different noise levels for objective and cost.

        Args:
            cost_function: Cost function of the problem. It can be passed as an argument or defined within.
            noise_std: Standard deviation of the observation noise (noise is added to both objective and cost).
            negate: If True, negates the objective function (but not the cost function).
        """
        self.cost_function = cost_function
        super().__init__(noise_std=noise_std, negate=negate)

        def evaluate_objective_true(self, X: Tensor) -> Tensor:
            r"""Evaluate the objective function (w/o observation noise) on a set of points."""
            raise NotImplementedError

        def evaluate_cost_true(self, X: Tensor) -> Tensor:
            r"""Evaluate the cost function (w/o observation noise) on a set of points."""
            raise NotImplementedError

        def evaluate_true(self, X: Tensor) -> Tensor:
            r"""Evaluate both the objective and cost functions (w/o observation noise) on a set of points."""
            raise NotImplementedError

        def get_cost_function(self, X: Tensor) -> Tensor:
            r"""Evaluate the cost function (w/o observation noise) on a set of points."""
            raise NotImplementedError


class CostAwareRosenbrock(CostAwareTestProblem):
    r"""
    f(x) = Rosenbrock(x).
    c(x) =  exp(1 - (k / (9 * dim)) * (\sum_i (x_i + 1)^2))
    """

    def __init__(
        self,
        dim: int = 3,
        seed: Optional[int] = None,
        cost_function: Optional[Callable] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self._bounds = [(-2.0, 2.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        rosenbrock = Rosenbrock(dim=self.dim)
        fX = rosenbrock.evaluate_true(X)
        return fX

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:

            if self.seed is None:
                X_plus_1_squared = torch.square(X + 1.0)
                ln_cX = 1.0 - (1.0 / (9.0 * self.dim)) * X_plus_1_squared.sum(dim=-1)
            else:
                a, b, c = get_cost_function_parameters(seed=self.seed)
                a = a.to(device=X.device, dtype=X.dtype)
                b = b.to(device=X.device, dtype=X.dtype)
                c = c.to(device=X.device, dtype=X.dtype)
                ln_cX = a * torch.cos(b * (2 * pi) * (X - 1.0 + c)).mean(dim=-1)
            cX = torch.exp(ln_cX)
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class CostAwareAckley(CostAwareTestProblem):
    r"""
    f(x) = Ackley(x)
    c(x) =  exp(1 - (1 / dim) * (sum_i x_i^2))
    """

    def __init__(
        self,
        dim: int = 5,
        seed: Optional[int] = None,
        cost_function: Optional[Callable] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self._bounds = [(-1.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        ackley = Ackley(dim=self.dim)
        fX = ackley.evaluate_true(X)
        return fX

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:
            if self.seed is None:
                ln_cX = 1.0 - (1.0 / self.dim) * torch.square(X).sum(dim=-1)
            else:
                a, b, c = get_cost_function_parameters(seed=self.seed)
                a = a.to(device=X.device, dtype=X.dtype)
                b = b.to(device=X.device, dtype=X.dtype)
                c = c.to(device=X.device, dtype=X.dtype)
                ln_cX = a * torch.cos(b * (2 * pi) * (X + c)).mean(dim=-1)

            cX = torch.exp(ln_cX)
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class CostAwareDropWave(CostAwareTestProblem):
    r"""
    f(x) = DropWave(x)
    c(x) =  exp(1 - (\sum_i x_i^2) / (5.12^2 * 2))
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        cost_function: Optional[Callable] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = 2
        self.seed = seed
        self._bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        super().__init__(
            cost_function=cost_function, noise_std=noise_std, negate=negate
        )

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        dropwave = DropWave()
        fX = dropwave.evaluate_true(X)
        return fX

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:
            if self.seed is None:
                ln_cX = 1.0 - torch.square(X).sum(dim=-1) / (5.12 ** 2 * self.dim)
            else:
                a, b, c = get_cost_function_parameters(seed=self.seed)
                a = a.to(device=X.device, dtype=X.dtype)
                b = b.to(device=X.device, dtype=X.dtype)
                c = c.to(device=X.device, dtype=X.dtype)
                ln_cX = a * torch.cos(b * (2 * pi / 5.12) * (X + c)).mean(dim=-1)
            cX = torch.exp(ln_cX)
        return cX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class CostAwareAugmentedRosenbrock(CostAwareTestProblem):
    r"""Augmented Rosenbrock function for cost-aware multi-fidelity optimization [Wu2019].

    f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2 + 0.1 * (1-x_{d-1}))^2 +
        (x_i - 1 + 0.1 * (1 - x_d)^2)^2)
    c(x) = 0.1 * exp{1 - ||x_opt - x_{1:d-1}||_2^2 / (d - 2)} + x_{d-1} * x_d
    """

    def __init__(
        self,
        cost_function: Optional[Callable] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:

        self.dim = 5
        self._bounds = [(-2.0, 2.0)] * 3 + [(0.0, 1.0)] * 2
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        X_curr = X[..., :-3]
        X_next = X[..., 1:-2]
        t1 = 100 * (X_next - X_curr ** 2 + 0.1 * (1 - X[..., -2:-1])) ** 2
        t2 = (X_curr - 1 + 0.1 * (1 - X[..., -1:]) ** 2) ** 2
        return (t1 + t2).sum(dim=-1)

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:
            aux = 1.0 - torch.square(X[..., :3] - 1.0).sum(dim=-1) / (
                9.0 * (self.dim - 2)
            )
            cX = 0.1 * torch.exp(aux) + X[..., -1] * X[..., -2]
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class CostAwareAugmentedHartmann(CostAwareTestProblem):
    r"""Augmented Hartmann function for cost-aware multi-fidelity optimization [Wu2019].

    The default cost function has been modified to make it dependent on all variables.

    f(x) = AugmentedHartmann(x)
    c(x) =  exp{1 - ||x_opt - x_{1:d-1}||_2^2 / (d - 1)} + x_d

    x_opt is the unique optimizer of the six-dimensional Hartmann function.
    """

    def __init__(
        self,
        cost_function: Optional[Callable] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:

        self.dim = 7
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self._optimizer = torch.tensor(
            [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
        )
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        augmented_hartmann = AugmentedHartmann()
        fX = augmented_hartmann.evaluate_true(X)
        return fX

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:
            aux = 1.0 - torch.square(X[..., :6] - self._optimizer).sum(dim=-1) / (
                self.dim - 1
            )
            cX = torch.exp(aux) + X[..., -1]
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class LeeEriksson(CostAwareTestProblem):
    r"""
    f(x) = ||x||_2 * sin(2 * pi * ||x||_2)
    c(x) = 2 - ||x||_2
    """

    def __init__(
        self,
        cost_function: Optional[Callable] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:

        self.dim = 3
        b = (0.9) / (2.0 * sqrt(self.dim))
        self._bounds = [(-b, b) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        norm_X = torch.norm(X, dim=-1)
        fX = norm_X * torch.sin(2 * pi * norm_X)
        return fX

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:
            cX = 2.0 - torch.norm(X, dim=-1)
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class CostAwareShekel(CostAwareTestProblem):
    r"""Shekel synthtetic test function.
    4-dimensional function (usually evaluated on `[0, 10]^4`):
        f(x) = -sum_{i=1}^10 (sum_{j=1}^4 (x_j - A_{ji})^2 + C_i)^{-1}
    f has one minimizer for its global minimum at `x^* = (4, 4, 4, 4)` with
    `f(x^*) = -10.5363`. The default cost function is an exponentiated
    quadratic with maximum value at global optimum og the objective function.
    """

    dim = 4
    _bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    _optimizers = [(4.0, 4.0, 4.0, 4.0)]

    def __init__(
        self,
        m: int = 5,
        seed: Optional[int] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.m = m
        self.seed = seed
        self.dim = 4
        self._bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
        self._optimizer = torch.tensor([4.0 for _ in range(4)])
        optvals = {5: -10.1532, 7: -10.4029, 10: -10.536443}
        self._optimal_value = optvals[self.m]
        super().__init__(noise_std=noise_std, negate=negate)

        self.register_buffer(
            "beta", torch.tensor([1, 2, 2, 4, 4, 6, 3, 7, 5, 5], dtype=torch.float)
        )
        C_t = torch.tensor(
            [
                [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            ],
            dtype=torch.float,
        )
        self.register_buffer("C", C_t.transpose(-1, -2))

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        beta = self.beta / 10.0
        result = -sum(
            1 / (torch.sum((X - self.C[i]) ** 2, dim=-1) + beta[i])
            for i in range(self.m)
        )
        return result

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:
            ln_cX = 1.0 - torch.square(X - self._optimizer).sum(dim=-1) / (
                (6.0 ** 2) * self.dim
            )
            if self.seed is None:
                ln_cX = 1.0 - torch.square(X - 4.0).sum(dim=-1) / (
                    (6.0 ** 2) * self.dim
                )
            else:
                a, b, c = get_cost_function_parameters(seed=self.seed)
                a = a.to(device=X.device, dtype=X.dtype)
                b = b.to(device=X.device, dtype=X.dtype)
                c = c.to(device=X.device, dtype=X.dtype)
                ln_cX = a * torch.cos(b * (2 * pi / 4.0) * (X - 4.0 + c)).mean(dim=-1)
            cX = torch.exp(ln_cX)
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class CostAwareHimmelblau(CostAwareTestProblem):
    r"""Himmelblau synthtetic test function.
    f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    c(x,y) = exp{(x^2 + y^2) / 25}
    """

    def __init__(
        self, epsilon=0.1, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.epsilon = epsilon
        self.dim = 2
        self._bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        objective_val = (
            (X[..., 0] ** 2 + X[..., 1] - 11.0) ** 2
            + (X[..., 0] + X[..., 1] ** 2 - 7.0) ** 2
            + self.epsilon
            * (torch.norm(X - torch.tensor([3.0, 2.0]), dim=-1) / sqrt(113.0))
        )
        return objective_val

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:
            ln_cX = 1.0 - 2.0 * (
                torch.norm(X - torch.tensor([3.0, 2.0]), dim=-1) / sqrt(113.0)
            )
            cX = torch.exp(ln_cX)
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class CostAwareRastrigin(CostAwareTestProblem):
    r"""Rastringin synthtetic test function.

    f(x) = 10.* dim + sum_{i=1}^{d} (x_i ** 2 - 10 * cos(2 * pi * x_i))
    c(x) = exp{(sum_{i=1}^{d} x_i + 5.12) /(5.12 * dim)}
    """

    def __init__(
        self,
        dim=4,
        seed: Optional[int] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self._bounds = [(-5.12, 5.12) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        objective_val = 10.0 * self.dim + torch.sum(
            X ** 2 - 10.0 * torch.cos(2.0 * pi * X), dim=-1
        )
        return objective_val

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:
            ln_cX = (X + 5.12).sum(dim=-1) / (self.dim * 5.12)
            if self.seed is None:
                ln_cX = (X + 5.12).sum(dim=-1) / (self.dim * 5.12)
            else:
                a, b, c = get_cost_function_parameters(seed=self.seed)
                a = a.to(device=X.device, dtype=X.dtype)
                b = b.to(device=X.device, dtype=X.dtype)
                c = c.to(device=X.device, dtype=X.dtype)
                ln_cX = a * torch.cos(b * (2 * pi / 5.12) * (X + c)).mean(dim=-1)

            cX = torch.exp(ln_cX)
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class CostAwareBukin(CostAwareTestProblem):
    r"""Bukin synthetic test function.

    f(x, y) = 100 * sqrt(|y - 0.01 * x^2|) + 0.01 * |x + 10|
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        cost_function: Optional[Callable] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = 2
        self.seed = seed
        self._bounds = [(-15.0, -5.0), (-3.0, 3.0)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        bukin = Bukin()
        fX = bukin.evaluate_true(X)
        return fX

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:

            if self.seed is not None:
                a, b, c = get_cost_function_parameters(seed=self.seed)
                a = a.to(device=X.device, dtype=X.dtype)
                b = b.to(device=X.device, dtype=X.dtype)
                c = c.to(device=X.device, dtype=X.dtype)
                ln_cX = (a / 2.0) * (
                    torch.cos(b * (2 * pi / 5.0) * (X[..., 0] + 5.0 + c))
                    + torch.cos(b * (2 * pi / 2.0) * (X[..., 1] - 1.0 + c))
                )
            cX = torch.exp(ln_cX)
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class CostAwareShaffer2(CostAwareTestProblem):
    r"""Shaffer2 synthetic test funciton.

    f(x, y) = (sin(x^2 - y^2) - 0.5) / (1 + 1e-3(x^2 + y^2))^2
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        cost_function: Optional[Callable] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = 2
        self.seed = seed
        self._bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        num = torch.sin(X[..., 0] ** 2 - X[..., 1] ** 2) ** 2 - 0.5
        den = (1.0 + 0.001 * (X[..., 0] ** 2 + X[..., 1] ** 2)) ** 2
        fX = 0.5 + (num / den)
        return fX

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:

            if self.seed is not None:
                a, b, c = get_cost_function_parameters(seed=self.seed)
                a = a.to(device=X.device, dtype=X.dtype)
                b = b.to(device=X.device, dtype=X.dtype)
                c = c.to(device=X.device, dtype=X.dtype)
                ln_cX = a * torch.cos(b * (2 * pi / 5.0) * (X + c)).mean(dim=-1)
            cX = torch.exp(ln_cX)
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class CostAwareGP(CostAwareTestProblem):
    r"""Cost-aware GP test problem.

    The objective and log-cost are (approximate)
    samples from independent GP prior distributions.
    """

    def __init__(
        self,
        dim: int,
        seed: int,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

        self.gp_draw = gp_posterior_mean_draw(
            input_dim=dim, output_dim=2, n_fantasies=100 * dim, seed=seed
        )

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        objective_val = self.gp_draw(X)[..., 0]
        return objective_val

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:
            ln_cX = self.gp_draw(X)[..., 1]
            cX = torch.exp(ln_cX)
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


class BanditsUCB(CostAwareTestProblem):
    r"""Non-stationary UCB bandit cost-aware test problem."""

    def __init__(
        self,
        n_reps: int,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = 3
        self.n_reps = n_reps
        self._bounds = [(0.0, 0.75) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_objective_true(self, X: Tensor) -> Tensor:
        objective_vals = []
        for i in range(X.shape[0]):
            objective_vals.append(
                bandits_ucb_simulator(cs=X[i, ...], n_reps=self.n_reps)[0]
            )
        objective_vals = torch.tensor(objective_vals)
        return objective_vals

    def evaluate_cost_true(self, X: Tensor) -> Tensor:
        if self.cost_function is not None:
            cX = self.cost_function(X)
        else:
            cost_vals = []
            for i in range(X.shape[0]):
                cost_vals.append(
                    bandits_ucb_simulator(cs=X[i, ...], n_reps=self.n_reps)[1]
                )
            cX = torch.tensor(cost_vals)
        return cX

    def evaluate_true(self, X: Tensor) -> Tensor:
        fX = self.evaluate_objective_true(X)
        return fX

    def get_cost_function(self) -> Callable:
        if self.cost_function is not None:
            cost_function = self.cost_function
        else:

            def cost_function(X):
                return self.evaluate_cost_true(X)

        return cost_function


def get_cost_function_parameters(seed):
    with manual_seed(seed=seed):
        a = torch.rand(1)
        a = 0.75 * a + 0.75

        b = torch.rand(1)
        b = 2.0 * b + 1.0

        c = torch.rand(1)
        c = 2 * pi * c
    return a, b, c
