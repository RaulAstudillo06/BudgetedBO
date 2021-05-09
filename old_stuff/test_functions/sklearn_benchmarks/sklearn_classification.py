#!/usr/bin/env python3
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from torch import Tensor

CLASSIFIER_PARAMS = {
    "RandomForest": ["n_estimators", "max_depth", "min_samples_split"],
    "DecisionTree": ["max_depth", "min_samples_split", "max_features"],
    "GradientBoosting": ["n_estimators", "min_samples_split", "learning_rate"],
    "Adaboost": ["n_estimators", "learning_rate"],
    "Bagging": ["n_estimators", "max_samples", "max_features"],
}

CLASSIFER_PARAM_BOUNDS = {
    "RandomForest": torch.tensor([[1.0, 1.0, -0.1], [256.0, 64.0, 0.0]]),
    "DecisionTree": torch.tensor([[1.0, -1.0, -3.0], [64.0, 0.0, np.log10(0.5)]]),
    "GradientBoosting": torch.tensor([[1.0, -1.0, -2.0], [256.0, 0.0, 0.0]]),
    "Adaboost": torch.tensor([[1.0, -2.0], [256.0, 0.0]]),
    "Bagging": torch.tensor([[1.0, -1.0, -2.0], [256.0, 0.0, 0.0]]),
}


def get_dataset(
    dataset_name: str,
    shuffle: bool = False,
    test_size: float = 0.33,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray]:
    r"""Get a dataset for sklearn model."""

    if dataset_name == "BreastCancer":
        data, targets = load_breast_cancer(return_X_y=True)
    elif dataset_name == "Digits":
        data, targets = load_digits(return_X_y=True)
    else:
        raise ValueError("Dataset name not recognized.")

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        targets,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


def build_params(model_name: str, X: np.ndarray) -> Dict:
    r"""Convert array to parameter dict."""
    if model_name == "RandomForest":
        params = {
            "n_estimators": np.rint(X[0]).astype(int),
            "max_depth": np.rint(X[1]).astype(int),
            "min_samples_split": float(10 ** X[2]),
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
    else:
        raise ValueError("Model name not recognized.")
    return params


def run_classifier(
    model_name: str,
    params: Dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    r"""Run the classifier and return score and cost."""
    name_to_cls = {
        "RandomForest": RandomForestClassifier,
        "DecisionTree": DecisionTreeClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "Adaboost": AdaBoostClassifier,
        "Bagging": BaggingClassifier,
    }
    model_with_default_params = name_to_cls[model_name]()
    model = name_to_cls[model_name](**params)

    # get baseline cost
    start_time = time.time()
    model_with_default_params.fit(X_train, y_train)
    elapsed_time_default = time.time() - start_time

    # run with real parameters
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score, elapsed_time / elapsed_time_default


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
        X_train, X_test, y_train, y_test = get_dataset(dataset_name=dataset_name)
        score, cost = run_classifier(
            model_name=model_name,
            params=params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        objs.append(score)
        costs.append(cost)

    return _tensorify(objs, batch).to(X), _tensorify(costs, batch).to(X)


def _tensorify(vals: List, batch: bool):
    return torch.tensor(vals) if batch else torch.tensor(vals).squeeze(0)
