#!/usr/bin/env python3

"""
Adopted from HPOlib2 (https://github.com/automl/HPOlib2)
"""

from typing import Dict, Optional, Union

# @dep=@/third-party:numpy:numpy-py
import numpy as np
from numpy.random import RandomState

# @dep=@/third-party:scikit-learn:scikit-learn-py
from sklearn.ensemble import RandomForestRegressor

from ..benchmark import HPOLibBenchmark
from ..utils import rng_helper
from ..utils.load import load_surrogate


class Surrogate2DWideResNet(HPOLibBenchmark):

    obj_path = "tree/model/automl/b2e1ec8b-6b1e-11eb-96d0-4857dd088237"

    def __init__(self, rng: Optional[Union[int, RandomState]] = None) -> None:
        self.rng = rng_helper.create_rng(rng)
        self._surrogate_objective: Optional[RandomForestRegressor] = None

    @property
    def surrogate_objective(self):
        if self._surrogate_objective is None:
            self._surrogate_objective = load_surrogate(path=self.obj_path)
        return self._surrogate_objective

    def objective_function(
        self, x: np.ndarray, budget: int = 4000, **kwargs
    ) -> Dict[str, float]:
        x_ = np.append(x, budget)[None, :]
        y = self.surrogate_objective.predict(x_)[0]
        return {"function_value": (1 - y / 100.0), "cost": budget}

    def objective_function_test(self, x: np.ndarray, **kwargs) -> Dict[str, float]:
        budget = 4000
        x_ = np.append(x, budget)[None, :]
        y = self.surrogate_objective.predict(x_)[0]
        return {"function_value": (1 - y / 100.0), "cost": budget}

    @staticmethod
    def get_meta_information():
        return {
            "name": (
                "Wide-ResNet with cosine annealing of the learning rate and "
                "a fix width of 32"
            ),
            "bounds": [[-2, 0], [8, 26]],  # log10 initial_lr  # depth
        }
