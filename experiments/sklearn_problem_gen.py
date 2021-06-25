#!/usr/bin/env python3
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_boston,
    load_diabetes,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from torch import Tensor


CLASSIFER_PARAM_BOUNDS = {
    "RandomForest": torch.tensor([[1.0, 1.0, -1.0], [256.0, 64.0, 0.0]]),
    "RandomForestReg": torch.tensor([[1.0, 1.0, -1.0], [256.0, 64.0, 0.0]]),
    "DecisionTree": torch.tensor([[1.0, -1.0, -1.0], [64.0, 0.0, 0.0]]),
    "GradientBoosting": torch.tensor([[1.0, -1.0, -2.0], [256.0, 0.0, 0.0]]),
    "Adaboost": torch.tensor([[1.0, -2.0], [256.0, 0.0]]),
    "Bagging": torch.tensor([[1.0, -1.0, -2.0], [256.0, 0.0, 0.0]]),
    "MLP": torch.tensor([[10.0, -5.0, -4.0, 50.0], [200.0, -3.0, -2.0, 300.0]]),
    "MLPReg": torch.tensor([[10.0, -5.0, -4.0, 50.0], [200.0, -3.0, -2.0, 300.0]]),
}


def build_params(model_name: str, X: np.ndarray) -> Dict:
    r"""Convert array to parameter dict."""

    if model_name == "RandomForest" or model_name == "RandomForestReg":
        params = {
            "n_estimators": np.rint(X[0]).astype(int),
            "max_depth": np.rint(X[1]).astype(int),
            "max_features": float(10 ** X[2]),
        }
    elif model_name == "DecisionTree":
        params = {
            "max_depth": np.rint(X[0]).astype(int),
            "min_samples_split": float(10 ** X[1]),
            "max_features": float(10 ** X[2]),
        }
    elif model_name == "GradientBoosting":
        params = {
            "n_estimators": np.rint(X[0]).astype(int),
            "min_samples_split": float(10 ** X[1]),
            "learning_rate": float(10 ** X[2]),
        }
    elif model_name == "Adaboost":
        params = {
            "n_estimators": np.rint(X[0]).astype(int),
            "learning_rate": float(10 ** X[1]),
        }
    elif model_name == "Bagging":
        params = {
            "n_estimators": np.rint(X[0]).astype(int),
            "max_samples": float(10 ** X[1]),
            "max_features": float(10 ** X[2]),
        }
    elif model_name == "MLP" or model_name == "MLPReg":
        params = {
            "hidden_layer_sizes": (np.rint(X[0]).astype(int),),
            "alpha": float(10 ** X[1]),
            "learning_rate_init": float(10 ** X[2]),
            "max_iter": np.rint(X[3]).astype(int),
        }
    else:
        raise ValueError("Model name not recognized.")
    logging.info(params)
    return params


def get_dataset(
    dataset_name: str,
    shuffle: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray]:
    r"""Get a dataset for sklearn model."""

    if dataset_name == "BreastCancer":
        data, targets = load_breast_cancer(return_X_y=True)
    elif dataset_name == "Digits":
        data, targets = load_digits(return_X_y=True)
    elif dataset_name == "Iris":
        data, targets = load_iris(return_X_y=True)
    elif dataset_name == "Boston":
        data, targets = load_boston(return_X_y=True)
    elif dataset_name == "Diabetes":
        data, targets = load_diabetes(return_X_y=True)
    else:
        raise ValueError("Dataset name not recognized.")
    return data, targets


def run_classifier(
    model_name: str,
    params: Dict,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    r"""Run the classifier and return score and cost."""
    name_to_cls = {
        "RandomForest": RandomForestClassifier,
        "DecisionTree": DecisionTreeClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "Adaboost": AdaBoostClassifier,
        "Bagging": BaggingClassifier,
        "MLP": MLPClassifier,
        "RandomForestReg": RandomForestRegressor,
        "MLPReg": MLPRegressor,
    }
    model_with_default_params = name_to_cls[model_name]()
    model = name_to_cls[model_name](**params)
    # get baseline cost
    start_time = time.time()
    #for _ in range(5):
        #model_with_default_params.fit(X, y)
    #elapsed_time_default = time.time() - start_time
    # run with real parameters
    start_time = time.time()
    for _ in range(5):
        model.fit(X, y)
    elapsed_time = time.time() - start_time
    # cross val score
    cv_score = cross_val_score(model, X, y, cv=5).mean()
    return cv_score, elapsed_time / 5#elapsed_time_default


def sklearn_classifier_objective(
    X: Tensor,
    dataset_name: str,
    model_name: str,
):
    r"""Entry point from benchmarking setup."""
    batch = X.ndimension() > 1
    if batch:
        xs = [x.detach().cpu().numpy() for x in X]
    else:
        xs = [X.detach().cpu().numpy()]
    objs, costs = [], []
    for x in xs:
        params = build_params(model_name=model_name, X=x)
        data_X, data_y = get_dataset(dataset_name=dataset_name)
        score, cost = run_classifier(
            model_name=model_name, params=params, X=data_X, y=data_y
        )
        objs.append(score)
        costs.append(cost)
    return _tensorify(objs, batch).to(X), _tensorify(costs, batch).to(X)


def _tensorify(vals: List, batch: bool):
    return torch.tensor(vals) if batch else torch.tensor(vals).squeeze(0)