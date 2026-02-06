#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


@dataclass
class StackingConfig:
    params_stacked: tuple[dict[str, Any], ...]
    cv: int
    append_other_model_types: bool

    final_estimator: ClassifierMixin = LogisticRegression()


@dataclass
class WeightedAverageConfig:
    params_stacked: tuple[dict[str, Any], ...]
    cv: int
    n_max: int = 3


@dataclass
class BaggingConfig:
    n_estimators: int = 32
    max_samples: [float | int] = 2048
    max_features: [float | int] = 1.0
    bootstrap: bool = True
    bootstrap_features: bool = False
