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
    "@INPROCEEDINGS{klein-ejs17,"
    "author = {A. Klein and S. Falkner and S. Bartels and P. Hennig and F. Hutter},"
    "title = {Fast Bayesian hyperparameter optimization on large datasets},"
    "booktitle = {Electronic Journal of Statistics},"
    "year = {2017},"
    "volume = {11}"
]


class SurrogateSVM(HPOLibBenchmark):

    obj_path = "tree/model/automl/7015b930-605a-11eb-882c-4857dd088237"
    cost_path = "tree/model/automl/74496bbd-605a-11eb-85b5-4857dd088237"

    def __init__(self, rng: Optional[Union[int, RandomState]] = None) -> None:
        self.s_min = 100 / 50000
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
        self, x: np.ndarray, dataset_fraction: int = 4000, **kwargs
    ) -> Dict[str, float]:
        test_point = np.append(x, dataset_fraction)[None, :]
        y = self.surrogate_objective.predict(test_point)[0]
        c = self.surrogate_cost.predict(test_point)[0]
        return {"function_value": y, "cost": c}

    def objective_function_test(self, x: np.ndarray, **kwargs) -> Dict[str, float]:
        return self.objective_function(x, dataset_fraction=1, **kwargs)

    @staticmethod
    def get_meta_information():
        return {
            "name": "Support Vector Machine",
            "bounds": [[-10, 10], [-10, 10]],  # C  # gamma
            "references": REFERENCES,
        }
