from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class NeuronState:
    """Holds mutable state for each neuron."""

    potentials: List[float]
    refractory: List[int]
    spikes: List[int]

    @classmethod
    def zeros(cls, size: int, v_reset: float = 0.0) -> "NeuronState":
        return cls([v_reset] * size, [0] * size, [0] * size)


@dataclass(frozen=True)
class SynapseCSR:
    """Read-only CSR representation storing pre-synaptic indices and delays."""

    indptr: Tuple[int, ...]
    indices: Tuple[int, ...]
    delays: Tuple[int, ...]
    size: int

    @classmethod
    def from_incoming(
        cls, adjacency: Sequence[Sequence[Tuple[int, int]]], size: int
    ) -> "SynapseCSR":
        if len(adjacency) != size:
            raise ValueError("Adjacency must provide one entry per neuron.")

        indptr = [0]
        indices: List[int] = []
        delays: List[int] = []
        for row in range(size):
            for pre, delay in adjacency[row]:
                if delay < 1:
                    raise ValueError("Synaptic delays must be at least 1 time step.")
                indices.append(pre)
                delays.append(delay)
            indptr.append(len(indices))
        return cls(tuple(indptr), tuple(indices), tuple(delays), size)

    def iter_row(self, row: int) -> Iterable[Tuple[int, int]]:
        start = self.indptr[row]
        end = self.indptr[row + 1]
        for idx in range(start, end):
            yield self.indices[idx], self.delays[idx]

    def max_delay(self) -> int:
        return max(self.delays, default=0)


@dataclass(frozen=True)
class SynapseTopology:
    """Bundles excitatory/inhibitory incoming CSR views and out-neighbor lists."""

    exc_in: SynapseCSR
    inh_in: SynapseCSR
    out_neighbors: Tuple[Tuple[int, ...], ...]

    @property
    def num_neurons(self) -> int:
        return len(self.out_neighbors)

    @classmethod
    def from_incoming_lists(
        cls,
        exc_in: Sequence[Sequence[Tuple[int, int]]],
        inh_in: Sequence[Sequence[Tuple[int, int]]],
        num_neurons: int,
    ) -> "SynapseTopology":
        exc = SynapseCSR.from_incoming(exc_in, num_neurons)
        inh = SynapseCSR.from_incoming(inh_in, num_neurons)

        out_lists = [set() for _ in range(num_neurons)]
        for post in range(num_neurons):
            for pre, _ in exc.iter_row(post):
                out_lists[pre].add(post)
            for pre, _ in inh.iter_row(post):
                out_lists[pre].add(post)
        out_neighbors = tuple(tuple(sorted(lst)) for lst in out_lists)
        return cls(exc_in=exc, inh_in=inh, out_neighbors=out_neighbors)

    def max_delay(self) -> int:
        return max(self.exc_in.max_delay(), self.inh_in.max_delay())


class DelayBuffer:
    """Ring buffer tracking recent spikes for each neuron."""

    def __init__(self, num_neurons: int, max_delay: int):
        if max_delay < 0:
            raise ValueError("max_delay must be non-negative")
        width = max(1, max_delay)
        self._buffers = [
            deque([0] * width, maxlen=width) for _ in range(num_neurons)
        ]
        self.max_delay = max_delay

    def advance(self, spikes: Sequence[int]) -> None:
        for idx, fired in enumerate(spikes):
            self._buffers[idx].appendleft(1 if fired else 0)

    def fired(self, neuron: int, delay: int) -> bool:
        if delay < 1 or delay > self.max_delay:
            raise ValueError("Delay queries must be within [1, max_delay].")
        return bool(self._buffers[neuron][delay - 1])


@dataclass
class StepResult:
    """Container returned by EventDrivenLIF.step."""

    spikes: List[int]
    next_active: List[int]


class EventDrivenLIF:
    """Event-driven LIF simulator matching the README reference implementation."""

    def __init__(
        self,
        topology: SynapseTopology,
        *,
        theta: float,
        tau_ref: int,
        lambda_: float,
        v_reset: float = 0.0,
        dt_ms: float = 1.0,
    ):
        if theta <= 0:
            raise ValueError("theta must be positive")
        if tau_ref < 0:
            raise ValueError("tau_ref must be non-negative")
        if not (0.0 <= lambda_ <= 1.0):
            raise ValueError("lambda_ should be in [0, 1]")

        self.topology = topology
        self.theta = theta
        self.tau_ref = tau_ref
        self.lambda_ = lambda_
        self.v_reset = v_reset
        self.dt_ms = dt_ms
        self.num_neurons = topology.num_neurons

        max_delay = max(1, topology.max_delay())
        self.delay_buffer = DelayBuffer(self.num_neurons, max_delay)
        self.state = NeuronState.zeros(self.num_neurons, v_reset=v_reset)
        self.time_step = 0

    def seed_from_noise(
        self, *, rate_hz: float, rng: random.Random | None = None
    ) -> List[int]:
        """Draw a set of active neurons using a Poisson rate approximation."""
        if rate_hz < 0:
            raise ValueError("rate_hz must be non-negative")
        rng = rng or random.Random()
        prob = min(rate_hz * (self.dt_ms / 1000.0), 1.0)
        return [idx for idx in range(self.num_neurons) if rng.random() < prob]

    def step(self, active_set: Iterable[int]) -> StepResult:
        """Advance the simulator by one step using only the provided active neurons."""
        unique_active = sorted(set(active_set))
        spikes = [0] * self.num_neurons
        self.state.spikes = spikes
        next_active = set()

        for neuron in unique_active:
            self._update_neuron(neuron, next_active)

        self.delay_buffer.advance(spikes)
        self.time_step += 1
        return StepResult(spikes=spikes, next_active=sorted(next_active))

    def _update_neuron(self, neuron: int, next_active: set[int]) -> None:
        ref = self.state.refractory[neuron]
        if ref > 0:
            self.state.refractory[neuron] = ref - 1
            self.state.potentials[neuron] = self.v_reset
            return

        current = self._collect_current(neuron)
        v = self.lambda_ * self.state.potentials[neuron] + current
        if v >= self.theta:
            self.state.spikes[neuron] = 1
            self.state.potentials[neuron] = self.v_reset
            self.state.refractory[neuron] = self.tau_ref
            next_active.update(self.topology.out_neighbors[neuron])
        else:
            self.state.potentials[neuron] = v

    def _collect_current(self, neuron: int) -> float:
        current = 0.0
        for pre, delay in self.topology.exc_in.iter_row(neuron):
            if self.delay_buffer.fired(pre, delay):
                current += 1.0
        for pre, delay in self.topology.inh_in.iter_row(neuron):
            if self.delay_buffer.fired(pre, delay):
                current -= 1.0
        return current


def sigma_rule_threshold(beta: float, k_in: int, p: float, tau_m_ms: float, dt_ms: float) -> float:
    """Utility helper mirroring README's σ rule for θ."""
    if not 0 <= p <= 1:
        raise ValueError("Activity probability p must be within [0, 1]")
    if k_in <= 0:
        raise ValueError("k_in must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")

    lam = math.exp(-dt_ms / tau_m_ms)
    w_eff = 1.0 / (1.0 - lam)
    sigma_i = math.sqrt(k_in * p * (1.0 - p) * w_eff)
    return beta * sigma_i
