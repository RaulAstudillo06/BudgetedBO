#!/usr/bin/env python3

import typing  # noqa F401

from ax.fb.utils.storage.manifold import AEManifoldUseCase
from ax.fb.utils.storage.manifold_torch import AEManifoldTorchClient

# @dep=@/third-party:scikit-learn:scikit-learn-py
from sklearn.ensemble import RandomForestRegressor


def load_surrogate(path: str) -> RandomForestRegressor:
    client = AEManifoldTorchClient(use_case=AEManifoldUseCase.AUTOML_MODEL)
    model = client.torch_load(path)
    return model
