from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ReadoutConfig:
    window: int = 200
    early_exit: bool = True
    max_future_gain_ratio: float = 0.3
    # Optional WTA (lateral inhibition) settings
    wta: bool = False
    wta_threshold: int = 1
    wta_inhibit: int = 1


class Readout:
    """Sliding-window spike count readout with optional early-exit.

    - Channels are added via add_channel(name, neuron_ids)
    - step(spike_mask) updates per-channel counts in a rolling window
    - can_early_exit(budget_remaining) decides if current leader is safe
    - emit() returns current label, scores, and latency (first-winning step)

    Deterministic and vectorized over NumPy arrays.
    """

    def __init__(self, cfg: ReadoutConfig | None = None) -> None:
        self.cfg = cfg or ReadoutConfig()
        if self.cfg.window <= 0:
            raise ValueError("window must be positive")

        self._names: List[str] = []
        self._ids: List[np.ndarray] = []

        self._win = int(self.cfg.window)
        self._hist = None  # shape (C, window) int32
        self._ptr = 0
        self._counts = None  # shape (C,) int32 current sums
        self._step_idx = 0  # 1-based steps processed counter

        # For early-exit estimation: Sum of total channel spikes across steps
        self._total_channel_spikes_accum = 0

        # Latching for earliest-time-wins
        self._latched_idx: int | None = None
        self._latency: int | None = None
        # Optional RNG for tie-breaking (deterministic via SeedSpace-derived Stream)
        self._tie_rng: object | None = None

    def set_tie_rng(self, rng: object) -> None:
        """Set a deterministic RNG for tie-breaking among equal scores.

        Expected interface: has method randbelow(n:int)->int or randint(a,b)->int or uniform()->float.
        """
        self._tie_rng = rng

    # ------------- Public API -------------
    def add_channel(self, name: str, neuron_ids: np.ndarray) -> None:
        ids = np.asarray(neuron_ids, dtype=np.int32)
        self._names.append(str(name))
        self._ids.append(ids)
        self._ensure_buffers()

    def step(self, spike_mask: np.ndarray) -> None:
        if not self._names:
            return
        sm = np.asarray(spike_mask, dtype=bool)
        C = len(self._names)

        # Count spikes per channel for this step
        step_counts = np.zeros(C, dtype=np.int32)
        for c, ids in enumerate(self._ids):
            if ids.size == 0:
                continue
            step_counts[c] = int(np.count_nonzero(sm[ids]))

        # Optional WTA: channels with count >= threshold inhibit others in this step
        if self.cfg.wta:
            thr = int(self.cfg.wta_threshold)
            inh = int(self.cfg.wta_inhibit)
            winners = np.nonzero(step_counts >= thr)[0]
            if winners.size > 0 and inh > 0:
                # Total inhibition proportional to sum of winners' counts
                total_inh = int(step_counts[winners].sum()) * inh
                if total_inh > 0:
                    for c in range(C):
                        if c in winners:
                            continue
                        step_counts[c] = max(0, int(step_counts[c]) - total_inh)

        # Update ring buffer and counts
        old = self._hist[:, self._ptr].copy()
        self._hist[:, self._ptr] = step_counts
        self._counts = self._counts + (step_counts - old)
        self._ptr = (self._ptr + 1) % self._win
        self._step_idx += 1

        # Update stats for early-exit
        self._total_channel_spikes_accum += int(step_counts.sum())

        # Latch earliest-time-wins: if nothing latched yet and someone fired now
        if self._latched_idx is None:
            pos = np.nonzero(step_counts > 0)[0]
            if pos.size > 0:
                # Pick the channel with largest count this step; tie -> RNG if available else lowest index
                vals = step_counts[pos]
                top = int(vals.max())
                cand = pos[vals == top]
                if cand.size == 1 or self._tie_rng is None:
                    best = int(cand[0])
                else:
                    # Try randbelow, else randint, else uniform
                    r = self._tie_rng
                    j = None
                    if hasattr(r, "randbelow"):
                        j = int(r.randbelow(int(cand.size)))  # type: ignore[attr-defined]
                    elif hasattr(r, "randint"):
                        j = int(r.randint(0, int(cand.size)))  # exclusive upper not guaranteed; clamp below
                        if j >= int(cand.size):
                            j = int(cand.size) - 1
                    elif hasattr(r, "uniform"):
                        u = float(r.uniform())
                        j = int(u * int(cand.size))
                        if j >= int(cand.size):
                            j = int(cand.size) - 1
                    else:
                        j = 0
                    best = int(cand[j])
                self._latched_idx = best
                self._latency = self._step_idx  # 1-based steps

    def can_early_exit(self, budget_remaining: int) -> bool:
        if not self.cfg.early_exit:
            return False
        if not self._names:
            return False
        if self._latched_idx is not None:
            # We already have a committed winner; can exit
            return True

        # Ensure counts are available
        counts = self._counts if self._counts is not None else np.zeros(0, dtype=np.int32)
        if counts.size == 0:
            return False
        # Need two best to compute leading gap
        order = np.argsort(counts)
        best = counts[order[-1]]
        second = counts[order[-2]] if counts.size >= 2 else 0
        delta = int(best) - int(second)

        # Estimate expected future gain bound
        steps = max(1, self._step_idx)
        p_to_readout = self._total_channel_spikes_accum / float(steps)
        U = float(budget_remaining) * float(self.cfg.max_future_gain_ratio) * float(p_to_readout)
        return delta > U

    def emit(self) -> Dict[str, object]:
        C = len(self._names)
        if C == 0:
            return {"label": "", "scores": {}, "latency": -1}

        counts = self._counts.astype(int)

        if self._latched_idx is not None:
            label_idx = self._latched_idx
            latency = int(self._latency if self._latency is not None else -1)
        else:
            # Use RNG tie-break if provided
            top = int(counts.max()) if counts.size > 0 else 0
            cand = np.nonzero(counts == top)[0]
            if cand.size <= 1 or self._tie_rng is None:
                label_idx = int(cand[0]) if cand.size > 0 else 0
            else:
                r = self._tie_rng
                if hasattr(r, "randbelow"):
                    j = int(r.randbelow(int(cand.size)))  # type: ignore[attr-defined]
                elif hasattr(r, "randint"):
                    j = int(r.randint(0, int(cand.size)))
                    if j >= int(cand.size):
                        j = int(cand.size) - 1
                elif hasattr(r, "uniform"):
                    u = float(r.uniform())
                    j = int(u * int(cand.size))
                    if j >= int(cand.size):
                        j = int(cand.size) - 1
                else:
                    j = 0
                label_idx = int(cand[j]) if cand.size > 0 else 0
            latency = -1

        scores = {self._names[i]: int(counts[i]) for i in range(C)}
        return {"label": self._names[label_idx], "scores": scores, "latency": latency}

    # ------------- Internal -------------
    def _ensure_buffers(self) -> None:
        C = len(self._names)
        if self._hist is None:
            self._hist = np.zeros((C, self._win), dtype=np.int32)
            self._counts = np.zeros(C, dtype=np.int32)
            self._ptr = 0
            return

        # Expand buffers to accommodate new channels (append new rows)
        curC = self._hist.shape[0]
        if C > curC:
            add = C - curC
            self._hist = np.vstack([self._hist, np.zeros((add, self._win), dtype=np.int32)])
            self._counts = np.concatenate([self._counts, np.zeros(add, dtype=np.int32)])


__all__ = ["ReadoutConfig", "Readout"]


# ---- Optional sliding-window readout (minimal API) ----

@dataclass
class ReadoutState:
    """Sliding-window readout state.

    - counts: int32 per-class counts in the current window
    - step: total steps processed
    - stable_steps: consecutive steps where the same class remains on top
    """

    counts: np.ndarray
    step: int = 0
    stable_steps: int = 0
    _hist: np.ndarray | None = None  # ring buffer of last window classes (-1 means none)
    _ptr: int = 0
    _last_pred: int | None = None


@dataclass
class WindowReadoutConfig:
    """Config for simple window readout.

    - window_steps: window length (>=1)
    - threshold: halt threshold for max count
    - hold_steps: consecutive steps above threshold to halt
    """

    window_steps: int
    threshold: int
    hold_steps: int = 0


def init_state(num_classes: int, cfg: WindowReadoutConfig) -> ReadoutState:
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")
    if int(cfg.window_steps) <= 0:
        raise ValueError("window_steps must be >= 1")
    counts = np.zeros(int(num_classes), dtype=np.int32)
    hist = np.full(int(cfg.window_steps), -1, dtype=np.int32)
    return ReadoutState(counts=counts, step=0, stable_steps=0, _hist=hist, _ptr=0, _last_pred=None)


def update(state: ReadoutState, spike_class: int | None) -> ReadoutState:
    """Update state with optional spiking class using FIFO window.

    For simplicity, approximate window by exponentially decaying counts:
    counts = counts - floor(counts / window) + one_hot(spike_class)
    Deterministic and bounded.
    """
    counts = state.counts.astype(np.int32, copy=True)
    hist = state._hist
    if hist is None:
        raise ValueError("state not initialized with history buffer")
    ptr = int(state._ptr)
    # Remove leaving class
    old = int(hist[ptr])
    if old >= 0:
        counts[old] = max(0, int(counts[old]) - 1)
    # Insert new class
    if spike_class is None:
        hist[ptr] = -1
    else:
        k = int(spike_class)
        if k < 0 or k >= counts.size:
            raise ValueError("spike_class out of range")
        hist[ptr] = k
        counts[k] = int(counts[k]) + 1
    ptr = (ptr + 1) % hist.size
    # Update stable_steps based on current top
    pred = int(np.argmax(counts)) if int(counts.max()) > 0 else None
    if pred is not None and state._last_pred is not None and pred == int(state._last_pred):
        stable = int(state.stable_steps) + 1
    else:
        stable = 1 if pred is not None else 0
    return ReadoutState(counts=counts, step=int(state.step) + 1, stable_steps=stable, _hist=hist, _ptr=ptr, _last_pred=pred)


def decide(state: ReadoutState, cfg: WindowReadoutConfig) -> Tuple[int | None, bool]:
    if state.counts.size == 0:
        raise ValueError("empty counts")
    top = int(np.max(state.counts))
    pred = int(np.argmax(state.counts)) if top > 0 else None
    halted = False
    if pred is not None and top >= int(cfg.threshold):
        halted = True if int(cfg.hold_steps) <= 0 else (state.stable_steps >= int(cfg.hold_steps))
    return pred, halted


__all__.extend(["ReadoutState", "WindowReadoutConfig", "init_state", "update", "decide"])
