#!/usr/bin/env python3

from __future__ import annotations

import torch
from torch import Tensor

from ..benchmark import HPOLibBenchmark


def hpolib_cost(benchmark: HPOLibBenchmark, X: Tensor) -> Tensor:
    batch = X.dim() > 1
    if batch:
        xs = [x.detach().cpu().numpy() for x in X]
    else:
        xs = [X.detach().cpu().numpy()]

    costs = [benchmark.objective_function(x)["cost"] for x in xs]
    results = torch.tensor(costs).to(X)
    return results if batch else results.squeeze(0)
