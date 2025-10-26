from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from topology import TopologyParams
from topology.generator import SmallWorldSampler

Edge = Tuple[int, int]
Adjacency = List[List[Edge]]


@dataclass
class PlasticityConfig:
    target_exc: int
    target_inh: int
    prune_quota: float = 0.1
    stdp_window: int = 20
    usage_decay: float = 0.95

    @classmethod
    def from_topology_params(
        cls,
        params: TopologyParams,
        *,
        prune_quota: float,
        stdp_window: int,
        usage_decay: float = 0.95,
    ) -> "PlasticityConfig":
        exc, inh = params.split_ei()
        return cls(
            target_exc=exc,
            target_inh=inh,
            prune_quota=prune_quota,
            stdp_window=stdp_window,
            usage_decay=usage_decay,
        )


@dataclass
class EdgeUsageTracker:
    """Tracks usage / STDP scores for each edge."""

    decay: float
    data: Dict[Tuple[int, int, str], float] = field(default_factory=dict)

    def edge_score(self, post: int, pre: int, sign: str) -> float:
        return self.data.get((post, pre, sign), 0.0)

    def bump(self, post: int, pre: int, sign: str, value: float) -> None:
        key = (post, pre, sign)
        self.data[key] = self.data.get(key, 0.0) + value

    def decay_all(self) -> None:
        for key in list(self.data.keys()):
            self.data[key] *= self.decay
            if abs(self.data[key]) < 1e-6:
                del self.data[key]


class StructuralPlasticity:
    """Implements prune-grow-stabilize cycle described in the README."""

    def __init__(
        self,
        exc_in: Adjacency,
        inh_in: Adjacency,
        sampler: SmallWorldSampler,
        config: PlasticityConfig,
    ):
        if len(exc_in) != len(inh_in):
            raise ValueError("exc_in and inh_in must have same length")
        self.exc_in = exc_in
        self.inh_in = inh_in
        self.sampler = sampler
        self.config = config
        self.tracker = EdgeUsageTracker(decay=config.usage_decay)

    def note_stdp(self, pre: int, post: int, delta_t: int, *, excitatory: bool) -> None:
        if abs(delta_t) > self.config.stdp_window:
            return
        sign = "exc" if excitatory else "inh"
        direction = 1.0 if delta_t > 0 else -1.0
        self.tracker.bump(post, pre, sign, direction)

    def run_cycle(self) -> None:
        self.tracker.decay_all()
        for post in range(len(self.exc_in)):
            self._prune_and_regrow(post, self.exc_in, "exc", self.config.target_exc)
            self._prune_and_regrow(post, self.inh_in, "inh", self.config.target_inh)

    def _prune_and_regrow(
        self,
        post: int,
        bank: Adjacency,
        sign: str,
        target: int,
    ) -> None:
        edges = bank[post]
        if not edges and target == 0:
            return

        prune_quota = int(len(edges) * self.config.prune_quota)
        surplus = max(0, len(edges) - target)
        prune_total = max(prune_quota, surplus)
        if prune_total > 0:
            scored = [
                (self.tracker.edge_score(post, pre, sign), idx)
                for idx, (pre, _) in enumerate(edges)
            ]
            scored.sort(key=lambda item: item[0])
            to_remove = sorted(idx for _, idx in scored[:prune_total])
            for offset, idx in enumerate(to_remove):
                edges.pop(idx - offset)

        if len(edges) < target:
            needed = target - len(edges)
            forbidden = {pre for pre, _ in edges}
            new_edges = self.sampler.sample(
                post, needed, sign, forbidden=forbidden
            )
            edges.extend(new_edges)

    def adjacency(self) -> Tuple[Adjacency, Adjacency]:
        return self.exc_in, self.inh_in
