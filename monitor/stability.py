from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List


@dataclass
class StabilityMetrics:
    average_rate_hz: float
    branching_factor: float
    synchrony_index: float


class StabilityMonitor:
    """Tracks firing rate, branching factor, and synchrony heuristics."""

    def __init__(self, num_neurons: int, window: int = 100, dt_ms: float = 1.0):
        if num_neurons <= 0:
            raise ValueError("num_neurons must be positive")
        self.num_neurons = num_neurons
        self.window = window
        self.dt_ms = dt_ms
        self._rates: Deque[int] = deque(maxlen=window)
        self._branch: Deque[float] = deque(maxlen=window)
        self._spike_history: List[int] = []

    def record_step(self, spikes: Iterable[int], next_active: Iterable[int]) -> None:
        spike_count = sum(1 for s in spikes if s)
        self._rates.append(spike_count)
        parents = max(1, spike_count)
        child_count = len(list(next_active))
        self._branch.append(child_count / parents)
        self._spike_history.append(spike_count)
        if len(self._spike_history) > self.window:
            self._spike_history.pop(0)

    def metrics(self) -> StabilityMetrics:
        rate = (
            sum(self._rates) / len(self._rates)
            if self._rates
            else 0.0
        )
        avg_rate_hz = (rate / self.num_neurons) * (1000.0 / self.dt_ms)
        branching = (
            sum(self._branch) / len(self._branch) if self._branch else 0.0
        )
        synchrony = self._synchrony_index()
        return StabilityMetrics(
            average_rate_hz=avg_rate_hz,
            branching_factor=branching,
            synchrony_index=synchrony,
        )

    def _synchrony_index(self) -> float:
        if len(self._spike_history) < 2:
            return 0.0
        mean = sum(self._spike_history) / len(self._spike_history)
        if mean == 0:
            return 0.0
        variance = sum((x - mean) ** 2 for x in self._spike_history) / len(
            self._spike_history
        )
        return min(5.0, variance / mean)
