from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from core.simulator import SynapseTopology
from symbolicplastic_snn.utils.prng import Stream

Coords = Sequence[Tuple[float, float]]
Layers = Sequence[int]


def distance_metric(
    coord_i: Tuple[float, float],
    coord_j: Tuple[float, float],
    layer_i: int,
    layer_j: int,
    *,
    alpha_layer: float,
) -> float:
    """Hybrid geometric + layer distance used throughout the README."""
    dx = coord_i[0] - coord_j[0]
    dy = coord_i[1] - coord_j[1]
    spatial = math.sqrt(dx * dx + dy * dy)
    layer_term = alpha_layer * abs(layer_i - layer_j)
    return spatial + layer_term


@dataclass(frozen=True)
class TopologyParams:
    k_in: int
    ei_ratio: float = 1.0
    sigma: float = 1.0
    alpha_layer: float = 2.0
    long_range_ratio: float = 0.01
    delay_per_unit: float = 1.0

    def split_ei(self) -> Tuple[int, int]:
        if self.k_in <= 0:
            raise ValueError("k_in must be positive")
        if self.ei_ratio < 0:
            raise ValueError("ei_ratio must be non-negative")
        if self.ei_ratio == 0:
            return 0, self.k_in
        exc = int(round(self.k_in * (self.ei_ratio / (1.0 + self.ei_ratio))))
        exc = max(1, min(self.k_in - 1, exc)) if self.k_in > 1 else self.k_in
        inh = self.k_in - exc
        return exc, inh


class SmallWorldSampler:
    """Distance driven sampler that yields candidate edges per neuron."""

    def __init__(
        self,
        coords: Coords,
        layers: Layers,
        params: TopologyParams,
        *,
        rng: object | None = None,
    ):
        if len(coords) != len(layers):
            raise ValueError("coords and layers length must match")
        self.coords = list(coords)
        self.layers = list(layers)
        self.params = params
        self.rng = rng or Stream(0)
        self._candidates = self._build_candidate_table()

    def _build_candidate_table(self) -> List[List[Tuple[int, float, float]]]:
        table: List[List[Tuple[int, float, float]]] = [[] for _ in self.coords]
        for post in range(len(self.coords)):
            for pre in range(len(self.coords)):
                if pre == post:
                    continue
                delta = distance_metric(
                    self.coords[pre],
                    self.coords[post],
                    self.layers[pre],
                    self.layers[post],
                    alpha_layer=self.params.alpha_layer,
                )
                local = math.exp(-delta / max(1e-6, self.params.sigma))
                long_range = self.params.long_range_ratio
                weight = max(1e-6, local + long_range)
                table[post].append((pre, weight, delta))
        return table

    def sample(
        self,
        post: int,
        count: int,
        sign: str,
        *,
        forbidden: Iterable[int] | None = None,
    ) -> List[Tuple[int, int]]:
        """Sample `count` edges targeting `post`, returns (pre, delay)."""
        if count <= 0:
            return []
        candidates = [
            (pre, weight, distance)
            for pre, weight, distance in self._candidates[post]
            if forbidden is None or pre not in forbidden
        ]
        selected = []
        available = candidates.copy()
        for _ in range(min(count, len(available))):
            total = sum(weight for _, weight, _ in available)
            if total <= 0:
                break
            # RNG: support both Python random.Random and PRNG Stream
            if hasattr(self.rng, "random") and callable(getattr(self.rng, "random")):
                u = float(self.rng.random())
            elif hasattr(self.rng, "uniform") and callable(getattr(self.rng, "uniform")):
                u = float(self.rng.uniform())
            else:
                u = float(Stream(0).uniform())
            pick = u * total
            cumulative = 0.0
            for idx, (pre, weight, distance) in enumerate(available):
                cumulative += weight
                if cumulative >= pick:
                    selected.append((pre, self._delay_from_distance(distance)))
                    available.pop(idx)
                    break
        return selected

    def _delay_from_distance(self, distance: float) -> int:
        scaled = distance / max(1e-6, self.params.delay_per_unit)
        return max(1, int(round(scaled))) if scaled > 0 else 1

    def build_topology(self) -> SynapseTopology:
        return build_small_world_topology(
            self.coords, self.layers, self.params, rng=self.rng
        )


def build_small_world_topology(
    coords: Coords,
    layers: Layers,
    params: TopologyParams,
    *,
    rng: object | None = None,
) -> SynapseTopology:
    """Constructs a SynapseTopology following README heuristics."""
    sampler = SmallWorldSampler(coords, layers, params, rng=rng)
    exc_target, inh_target = params.split_ei()
    num_neurons = len(coords)
    exc_in = [[] for _ in range(num_neurons)]
    inh_in = [[] for _ in range(num_neurons)]

    for post in range(num_neurons):
        exc_samples = sampler.sample(post, exc_target, "exc", forbidden=None)
        inh_samples = sampler.sample(post, inh_target, "inh", forbidden=None)
        exc_in[post].extend(exc_samples)
        inh_in[post].extend(inh_samples)

    return SynapseTopology.from_incoming_lists(exc_in, inh_in, num_neurons=num_neurons)
