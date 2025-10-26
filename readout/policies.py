from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class WinnerTakeAll:
    group_map: Sequence[int]

    def __post_init__(self) -> None:
        self.num_groups = max(self.group_map) + 1
        self.scores = [0] * self.num_groups

    def observe(self, spikes: Sequence[int]) -> None:
        for idx, fired in enumerate(spikes):
            if not fired:
                continue
            group = self.group_map[idx]
            self.scores[group] += 1

    def winner(self) -> int | None:
        if not any(self.scores):
            return None
        return max(range(self.num_groups), key=lambda g: self.scores[g])

    def reset(self) -> None:
        self.scores = [0] * self.num_groups


class PerceptronReadout:
    """Perceptron style readout without backprop."""

    def __init__(self, n_features: int, n_classes: int, lr: float = 1.0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.bias = [0.0] * n_classes

    def predict(self, features: Sequence[float]) -> int:
        scores = [
            self.bias[c]
            + sum(w * x for w, x in zip(self.weights[c], features))
            for c in range(self.n_classes)
        ]
        return max(range(self.n_classes), key=lambda c: scores[c])

    def train_epoch(
        self, samples: Sequence[Sequence[float]], labels: Sequence[int]
    ) -> None:
        for features, label in zip(samples, labels):
            pred = self.predict(features)
            if pred == label:
                continue
            self._update(pred, label, features)

    def _update(self, wrong: int, correct: int, features: Sequence[float]) -> None:
        for idx, value in enumerate(features):
            self.weights[correct][idx] += self.lr * value
            self.weights[wrong][idx] -= self.lr * value
        self.bias[correct] += self.lr
        self.bias[wrong] -= self.lr
