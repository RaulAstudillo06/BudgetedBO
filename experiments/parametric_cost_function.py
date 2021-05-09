import torch
from botorch.utils.sampling import manual_seed
from math import pi


def get_cost_function_parameters(seed):
    with manual_seed(seed=seed):
        a = torch.rand(1)
        a = 0.75 * a + 0.75

        b = torch.rand(1)
        b = 2.0 * b + 1.0

        c = torch.rand(1)
        c = 2 * pi * c
    return a, b, c