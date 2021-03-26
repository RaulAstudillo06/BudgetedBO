#!/usr/bin/env python3

"""
Adopted from HPOlib2 (https://github.com/automl/HPOlib2)
"""

from copy import deepcopy
from typing import Dict, Optional

# @dep=@/third-party:numpy:numpy-py
import numpy as np

# @dep=@/third-party:scikit-learn:scikit-learn-py
from sklearn.ensemble import RandomForestRegressor

from ..benchmark import HPOLibBenchmark
from ..utils.load import load_surrogate


class SurrogateParamNet(HPOLibBenchmark):

    obj_paths = {
        "adult": "tree/model/automl/848a6a03-5f30-11eb-ab9e-4857dd088237",
        "higgs": "tree/model/automl/87d7f2e7-5f30-11eb-893c-4857dd088237",
        "letter": "tree/model/automl/8aafd942-5f30-11eb-82cd-4857dd088237",
        "mnist": "tree/model/automl/8d90fd30-5f30-11eb-bf5a-4857dd088237",
        "optdigits": "tree/model/automl/90f2824e-5f30-11eb-9313-4857dd088237",
        "poker": "tree/model/automl/000ab007-5f36-11eb-93ad-4857dd088237",
    }
    cost_paths = {
        "adult": "tree/model/automl/87139919-5f30-11eb-ab5e-4857dd088237",
        "higgs": "tree/model/automl/89fffdfe-5f30-11eb-9e43-4857dd088237",
        "letter": "tree/model/automl/8cf4677d-5f30-11eb-8abe-4857dd088237",
        "mnist": "tree/model/automl/90228d47-5f30-11eb-be81-4857dd088237",
        "optdigits": "tree/model/automl/9331aa46-5f30-11eb-b121-4857dd088237",
        "poker": "tree/model/automl/026637e6-5f36-11eb-8099-4857dd088237",
    }

    def __init__(self, dataset: str, **kwargs) -> None:
        """
        Args:
            dataset: Dataset the network was trained on. Available: adult, higgs,
                letter, mnist, optdigits, poker
        """
        if dataset not in self.obj_paths:
            raise ValueError(f"No surrogate found for {dataset}")
        self.n_epochs: int = 50
        self.dataset: str = dataset
        self._surrogate_objective: Optional[RandomForestRegressor] = None
        self._surrogate_cost: Optional[RandomForestRegressor] = None

    @property
    def surrogate_objective(self):
        if self._surrogate_objective is None:
            self._surrogate_objective = load_surrogate(
                path=self.obj_paths[self.dataset]
            )
        return self._surrogate_objective

    @property
    def surrogate_cost(self):
        if self._surrogate_cost is None:
            self._surrogate_cost = load_surrogate(path=self.cost_paths[self.dataset])
        return self._surrogate_cost

    def get_empirical_f_opt(self):
        """
        Returns average across minimal value in all trees
        :return:
        """
        ms = []
        for t in self.surrogate_objective.estimators_:
            ms.append(np.min(t.tree_.value))
        return np.mean(ms)

    def objective_function(
        self, x: np.ndarray, step: Optional[int] = None, **kwargs
    ) -> Dict[str, float]:
        x_ = deepcopy(x)
        x_[0] = 10 ** x_[0]
        x_[1] = 2 ** x_[1]
        x_[2] = 2 ** x_[2]
        x_[3] = 10 ** x_[3]
        lc = self.surrogate_objective.predict(x_[None, :])[0]
        c = self.surrogate_cost.predict(x_[None, :])[0]

        if step is None:
            y = lc[-1]
            cost = c
        else:
            y = lc[step]
            cost = c / self.n_epochs * step

        return {"function_value": y, "cost": cost, "learning_curve": lc}

    def objective_function_test(self, x: np.ndarray, **kwargs) -> Dict[str, float]:
        x_ = deepcopy(x)
        x_[0] = 10 ** x_[0]
        x_[1] = 2 ** x_[1]
        x_[2] = 2 ** x_[2]
        x_[3] = 10 ** x_[3]
        lc = self.surrogate_objective.predict(x_[None, :])[0]
        c = self.surrogate_cost.predict(x_[None, :])[0]
        y = lc[-1]
        return {"function_value": y, "cost": c}

    @staticmethod
    def get_meta_information():
        return {
            "name": "Stefans reparameterization of paramnet",
            # 'bounds': [[1e-6, 1e-2],  # initial_lr
            #            [8, 256],  # batch_size
            #            [16, 256],  # average_units_per_layer
            #            [1e-4, 1],  # final_lr_fraction
            #            [0, 1.],  # shape_parameter_1
            #            [1, 5],  # num_layers
            #            [0, .5],  # dropout_0
            #            [0, .5]]  # dropout_1
            "bounds": [
                [-6, -2],  # log10 initial_lr
                [3, 8],  # log2 batch_size
                [4, 8],  # log2 average_units_per_layer
                [-4, 0],  # log10 final_lr_fraction
                [0, 1.0],  # shape_parameter_1
                [1, 5],  # num_layers
                [0, 0.5],  # dropout_0
                [0, 0.5],  # dropout_1
            ],
        }


class SurrogateReducedParamNetTime(SurrogateParamNet):
    def objective_function(
        self, x: np.ndarray, budget: Optional[float] = None, **kwargs
    ) -> Dict[str, float]:
        # If no budget is specified we train this config for the max number of epochs
        if budget is None:
            budget = np.inf

        x_ = np.zeros([1, 8], dtype=np.float)

        x_[0, 0] = 10 ** x[0]
        x_[0, 1] = 2 ** x[1]
        x_[0, 2] = 2 ** x[2]
        x_[0, 3] = 10 ** x[3]
        x_[0, 4] = 0.5
        x_[0, 5] = x[4]
        x_[0, 6] = x[5]
        x_[0, 7] = x[5]

        lc = self.surrogate_objective.predict(x_)[0]
        c = self.surrogate_cost.predict(x_)[0]

        # Check if we can afford a single epoch in the budget
        if c / self.n_epochs > budget:
            # TODO: Return random performance here instead
            y = 1
            return {"function_value": y, "cost": budget}

        learning_curves_cost = np.linspace(c / self.n_epochs, c, self.n_epochs)

        idx = np.where(learning_curves_cost < budget)[0][-1]
        y = lc[idx]

        return {"function_value": y, "cost": budget, "lc": lc[:idx].tolist()}

    def objective_function_test(self, x: np.ndarray, **kwargs) -> Dict[str, float]:
        x_ = np.zeros([1, 8], dtype=np.float)

        x_[0, 0] = 10 ** x[0]
        x_[0, 1] = 2 ** x[1]
        x_[0, 2] = 2 ** x[2]
        x_[0, 3] = 10 ** x[3]
        x_[0, 4] = 0.5
        x_[0, 5] = x[4]
        x_[0, 6] = x[5]
        x_[0, 7] = x[5]
        lc = self.surrogate_objective.predict(x_)[0]
        c = self.surrogate_cost.predict(x_)[0]
        y = lc[-1]
        return {"function_value": y, "cost": c}

    @staticmethod
    def get_meta_information():
        return {
            "name": "Stefans reparameterization of paramnet",
            # 'bounds': [[1e-6, 1e-2],  # initial_lr
            #            [8, 256],  # batch_size
            #            [16, 256],  # average_units_per_layer
            #            [1e-4, 1],  # final_lr_fraction
            #            [0, 1.],  # shape_parameter_1
            #            [1, 5],  # num_layers
            #            [0, .5],  # dropout_0
            #            [0, .5]]  # dropout_1
            "bounds": [
                [-6, -2],  # log10 initial_lr
                [3, 8],  # log2 batch_size
                [4, 8],  # log2 average_units_per_layer
                [-4, 0],  # log10 final_lr_fraction
                # [0, 1.],  # shape_parameter_1
                [1, 5],  # num_layers
                # [0, .5]  # dropout_0
                [0, 0.5],  # dropout_1
            ],
        }


class SurrogateParamNetTime(SurrogateParamNet):
    def objective_function(
        self, x: np.ndarray, budget: Optional[float] = None, **kwargs
    ) -> Dict[str, float]:
        # If no budget is specified we train this config for the max number of epochs
        if budget is None:
            return super().objective_function(x)

        x_ = deepcopy(x)
        x_[0] = 10 ** x_[0]
        x_[1] = 2 ** x_[1]
        x_[2] = 2 ** x_[2]
        x_[3] = 10 ** x_[3]
        lc = self.surrogate_objective.predict(x_[None, :])[0]
        c = self.surrogate_cost.predict(x_[None, :])[0]

        # Check if we can afford a single epoch in the budget
        if c / self.n_epochs > budget:
            # TODO: Return random performance here instead
            y = 1
            return {"function_value": y, "cost": budget}

        learning_curves_cost = np.linspace(c / self.n_epochs, c, self.n_epochs)

        if budget < c:
            idx = np.where(learning_curves_cost < budget)[0][-1]
            y = lc[idx]
            return {
                "function_value": y,
                "cost": budget,
                "learning_curve": lc[:idx],
                "observed_epochs": len(lc[:idx]),
            }
        else:
            # If the budget is larger than the actual runtime we extrapolate the
            # learning curve
            t_left = budget - c
            n_epochs = int(t_left / (c / self.n_epochs))
            lc = np.append(lc, np.ones(n_epochs) * lc[-1])
            y = lc[-1]
            return {
                "function_value": y,
                "cost": budget,
                "learning_curve": lc,
                "observed_epochs": len(lc),
            }
