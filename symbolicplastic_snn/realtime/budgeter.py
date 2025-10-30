from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EventBudget:
    """Simple per-step event budget tracker.

    - b_step: maximum allowed cost for the step
    - used: accumulated cost so far in the step

    Cost model: cost(event) = len(event.indices)
    """

    b_step: int
    used: int = 0

    def allow(self, cost: int) -> bool:
        """Return True if charging `cost` would not exceed the step budget."""
        if cost < 0:
            raise ValueError("cost must be non-negative")
        return (self.used + int(cost)) <= int(self.b_step)

    def charge(self, cost: int) -> None:
        """Consume `cost` from the remaining budget; must be pre-allowed."""
        if cost < 0:
            raise ValueError("cost must be non-negative")
        self.used += int(cost)


__all__ = ["EventBudget"]

