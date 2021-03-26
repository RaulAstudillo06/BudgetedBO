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


REFERENCES = [
    "@InProceedings{klein-iclr17,"
    "author = {A. Klein and S. Falkner and J. T. Springenberg and F. Hutter},"
    "title = {Learning Curve Prediction with {Bayesian} Neural Networks},"
    "booktitle = {International Conference on Learning Representations (ICLR)"
    " 2017 Conference Track},"
    "year = {2017},"
    "month = apr"
]


class SurrogateCNN(HPOLibBenchmark):

    obj_path = "tree/model/automl/4abfe646-5d03-11eb-8534-4857dd088237"
    cost_path = "tree/model/automl/90f87790-5d05-11eb-8246-4857dd088237"

    def __init__(self, rng: Optional[Union[int, RandomState]] = None) -> None:
        self.n_epochs: int = 40
        self.rng = rng_helper.create_rng(rng)
        self._surrogate_objective: Optional[RandomForestRegressor] = None
        self._surrogate_cost: Optional[RandomForestRegressor] = None

    @property
    def surrogate_objective(self):
        if self._surrogate_objective is None:
            self._surrogate_objective = load_surrogate(path=self.obj_path)
        return self._surrogate_objective

    @property
    def surrogate_cost(self):
        if self._surrogate_cost is None:
            self._surrogate_cost = load_surrogate(path=self.cost_path)
        return self._surrogate_cost

    def objective_function(
        self, x: np.ndarray, step: Optional[int] = None, **kwargs
    ) -> Dict[str, float]:
        if step is None:
            step = self.n_epochs
        test_point = np.concatenate((x[None, :], np.array([[step]])), axis=1)
        y = self.surrogate_objective.predict(test_point)[0]
        c = np.cumsum(
            [
                self.surrogate_cost.predict(
                    np.concatenate((x[None, :], np.array([[i]])), axis=1)
                )[0]
                for i in range(1, step + 1)
            ]
        )[0]
        return {"function_value": y, "cost": c}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x, step=self.n_epochs)

    @staticmethod
    def get_meta_information():
        return {
            "name": "Convolutional Neural Network Surrogate",
            "bounds": [
                [0, 1],  # init_learning_rate, [10**-6, 10**0]
                [0, 1],  # batch_size, [32, 512]
                [0, 1],  # n_units_1,  [2**4, 2**8]
                [0, 1],  # n_units_2,  [2**4, 2**8]
                [0, 1],  # n_units_3, [2**4, 2**8]
            ],
            "references": REFERENCES,
        }
