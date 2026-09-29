from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from river import forest, tree


RIVER_METHODS = {
    "adaptive_random_forest",
    "hoeffding_adaptive_tree",
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    total = float(exp.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(logits.shape, 1.0 / len(logits), dtype=np.float64)
    return exp / total


def _cross_entropy(probabilities: np.ndarray, y: int) -> float:
    return -float(np.log(np.clip(probabilities[int(y)], 1e-12, 1.0)))


@dataclass
class FESLDiagnostics:
    historical_weight: float
    adaptive_weight: float
    mapping_mse: float


class FESLClassifier:
    """Multiclass adaptation of the FESL-c weighted combination mechanism.

    The map from S2 to S1 is learned only from a supplied historical paired
    calibration prefix.  During S2, the historical and adaptive predictors are
    combined with multiplicative Hedge weights and then updated prequentially.
    """

    def __init__(
        self,
        dimension1: int,
        dimension2: int,
        num_classes: int,
        classifier_learning_rate: float = 0.15,
        hedge_beta: float = 0.95,
        ridge: float = 1e-4,
    ) -> None:
        self.dimension1 = int(dimension1)
        self.dimension2 = int(dimension2)
        self.num_classes = int(num_classes)
        self.classifier_learning_rate = float(classifier_learning_rate)
        self.hedge_beta = float(hedge_beta)
        self.ridge = float(ridge)
        self.old_weights = np.zeros((self.num_classes, self.dimension1), dtype=np.float64)
        self.old_bias = np.zeros(self.num_classes, dtype=np.float64)
        self.new_weights = np.zeros((self.num_classes, self.dimension2), dtype=np.float64)
        self.new_bias = np.zeros(self.num_classes, dtype=np.float64)
        self.mapper = np.zeros((self.dimension2, self.dimension1), dtype=np.float64)
        for index in range(min(self.dimension1, self.dimension2)):
            self.mapper[index, index] = 1.0
        self.expert_weights = np.array([0.5, 0.5], dtype=np.float64)
        self.old_step = 0
        self.new_step = 0
        self.mapping_mse = float("nan")

    def fit_mapping(self, paired_s1: np.ndarray, paired_s2: np.ndarray) -> None:
        old = np.asarray(paired_s1, dtype=np.float64)
        new = np.asarray(paired_s2, dtype=np.float64)
        if old.ndim != 2 or new.ndim != 2 or len(old) != len(new) or len(old) == 0:
            raise ValueError("FESL mapping calibration requires non-empty paired matrices")
        gram = new.T @ new + self.ridge * np.eye(new.shape[1])
        self.mapper = np.linalg.solve(gram, new.T @ old)
        reconstruction = new @ self.mapper
        self.mapping_mse = float(np.mean((reconstruction - old) ** 2))

    @staticmethod
    def _update_softmax(
        weights: np.ndarray,
        bias: np.ndarray,
        x: np.ndarray,
        y: int,
        eta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities = _softmax(weights @ x + bias)
        target = np.zeros(len(probabilities), dtype=np.float64)
        target[int(y)] = 1.0
        weights -= eta * (probabilities - target)[:, None] * x[None, :]
        bias -= eta * (probabilities - target)
        return weights, bias

    def predict_s1(self, x1: np.ndarray) -> np.ndarray:
        x1 = np.asarray(x1, dtype=np.float64).reshape(-1)
        return _softmax(self.old_weights @ x1 + self.old_bias)

    def learn_s1(self, x1: np.ndarray, y: int) -> None:
        x1 = np.asarray(x1, dtype=np.float64).reshape(-1)
        eta = self.classifier_learning_rate / np.sqrt(self.old_step + 1.0)
        self.old_weights, self.old_bias = self._update_softmax(
            self.old_weights, self.old_bias, x1, int(y), eta
        )
        self.old_step += 1

    def predict_s2(self, x2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x2 = np.asarray(x2, dtype=np.float64).reshape(-1)
        mapped = x2 @ self.mapper
        historical = _softmax(self.old_weights @ mapped + self.old_bias)
        adaptive = _softmax(self.new_weights @ x2 + self.new_bias)
        fused = self.expert_weights[0] * historical + self.expert_weights[1] * adaptive
        fused /= max(float(fused.sum()), 1e-12)
        return fused, historical, adaptive

    def learn_s2(
        self,
        x2: np.ndarray,
        y: int,
        historical: Optional[np.ndarray] = None,
        adaptive: Optional[np.ndarray] = None,
    ) -> None:
        x2 = np.asarray(x2, dtype=np.float64).reshape(-1)
        if historical is None or adaptive is None:
            _, historical, adaptive = self.predict_s2(x2)
        losses = np.array(
            [_cross_entropy(historical, int(y)), _cross_entropy(adaptive, int(y))]
        )
        self.expert_weights *= np.power(self.hedge_beta, losses)
        self.expert_weights /= max(float(self.expert_weights.sum()), 1e-12)
        eta = self.classifier_learning_rate / np.sqrt(self.new_step + 1.0)
        # FESL-c updates the historical classifier on recovered old features
        # as well as the new classifier (paper Algorithm 2, step 8).
        self.old_weights, self.old_bias = self._update_softmax(
            self.old_weights, self.old_bias, x2 @ self.mapper, int(y), eta
        )
        self.new_weights, self.new_bias = self._update_softmax(
            self.new_weights, self.new_bias, x2, int(y), eta
        )
        self.new_step += 1

    def diagnostics(self) -> FESLDiagnostics:
        return FESLDiagnostics(
            historical_weight=float(self.expert_weights[0]),
            adaptive_weight=float(self.expert_weights[1]),
            mapping_mse=float(self.mapping_mse),
        )


def make_baseline(name, dimension1, dimension2, original_dimension, num_classes, seed, **options):
    """Create any baseline used in the thesis experiments."""
    if name == "old3s":
        from old3s_baseline import OLD3SBaseline
        return OLD3SBaseline(dimension1, dimension2, num_classes, seed)
    if name == "fesl":
        return FESLClassifier(
            dimension1,
            dimension2,
            num_classes,
            classifier_learning_rate=options.get("fesl_learning_rate", 0.15),
            ridge=options.get("fesl_mapper_ridge", 1e-4),
            hedge_beta=options.get("fesl_hedge_beta", 0.95),
        )

    river_options = dict(options.get("river_options", {}))
    if name == "adaptive_random_forest":
        return forest.ARFClassifier(seed=seed, **{"n_models": 10, **river_options})
    if name == "hoeffding_adaptive_tree":
        return tree.HoeffdingAdaptiveTreeClassifier(seed=seed, **river_options)
    raise ValueError(f"Unknown baseline: {name}")
