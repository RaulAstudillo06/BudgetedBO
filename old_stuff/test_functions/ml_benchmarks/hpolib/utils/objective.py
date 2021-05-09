#!/usr/bin/env python3

import torch
from torch import Tensor

from ..benchmark import HPOLibBenchmark


def hpolib_objective(benchmark: HPOLibBenchmark, X: Tensor) -> Tensor:
    batch = X.dim() > 1
    if batch:
        xs = [x.detach().cpu().numpy() for x in X]
    else:
        xs = [X.detach().cpu().numpy()]
    # If necessary, could run these as async workflows...
    vals = [benchmark.objective_function(x)["function_value"] for x in xs]
    # HPOLib convention is to minimize, so return negative objectives
    results = -torch.tensor(vals).to(X)
    return results if batch else results.squeeze(0)
