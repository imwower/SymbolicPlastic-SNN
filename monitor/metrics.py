from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Metrics:
    """Lightweight run-time metrics aggregator.

    Tracks counts deterministically and supports reset/merge/snapshot.
    """

    # Core counters
    steps: int = 0
    events_generated: int = 0
    events_processed: int = 0
    events_deferred: int = 0
    budget_rejects: int = 0
    early_exits: int = 0

    # Readout winners per channel label (by name)
    winners: Dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.steps = 0
        self.events_generated = 0
        self.events_processed = 0
        self.events_deferred = 0
        self.budget_rejects = 0
        self.early_exits = 0
        self.winners.clear()

    def add_winner(self, label: str) -> None:
        self.winners[str(label)] = int(self.winners.get(str(label), 0)) + 1

    def merge(self, other: "Metrics") -> None:
        self.steps += int(other.steps)
        self.events_generated += int(other.events_generated)
        self.events_processed += int(other.events_processed)
        self.events_deferred += int(other.events_deferred)
        self.budget_rejects += int(other.budget_rejects)
        self.early_exits += int(other.early_exits)
        for k, v in other.winners.items():
            self.winners[str(k)] = int(self.winners.get(str(k), 0)) + int(v)

    def to_dict(self) -> Dict[str, int | Dict[str, int]]:
        return {
            "steps": int(self.steps),
            "events_generated": int(self.events_generated),
            "events_processed": int(self.events_processed),
            "events_deferred": int(self.events_deferred),
            "budget_rejects": int(self.budget_rejects),
            "early_exits": int(self.early_exits),
            "winners": {str(k): int(v) for k, v in sorted(self.winners.items())},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int | Dict[str, int]]) -> "Metrics":
        m = cls()
        m.steps = int(d.get("steps", 0))
        m.events_generated = int(d.get("events_generated", 0))
        m.events_processed = int(d.get("events_processed", 0))
        m.events_deferred = int(d.get("events_deferred", 0))
        m.budget_rejects = int(d.get("budget_rejects", 0))
        m.early_exits = int(d.get("early_exits", 0))
        winners = d.get("winners", {}) or {}
        if isinstance(winners, dict):
            for k, v in winners.items():
                m.winners[str(k)] = int(v)
        return m

__all__ = ["Metrics"]

