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

    @property
    def remaining(self) -> int:
        """Return remaining budget for this step (non-negative)."""
        rem = int(self.b_step) - int(self.used)
        return rem if rem >= 0 else 0


__all__ = ["EventBudget"]


# ---- Optional simple budgeter with carry ----

@dataclass
class Budgeter:
    """Simple per-step budget with carry and saturation.

    - per_step: maximum tokens that can be spent per step.
    - carry: unspent tokens carried to next step (capped at per_step).
    """

    per_step: int
    carry: int = 0


def admit(b: Budgeter, demand: int) -> tuple[int, Budgeter]:
    """Admit up to available tokens for this step given a demand.

    Returns (granted, new_budgeter). Carry saturates at per_step.
    """
    avail = int(b.per_step) + max(0, int(b.carry))
    need = max(0, int(demand))
    grant = min(avail, need)
    remaining = avail - grant
    # Carry is what remains, but not exceeding per_step
    new_carry = min(int(b.per_step), remaining)
    return grant, Budgeter(per_step=int(b.per_step), carry=new_carry)


def reset(b: Budgeter) -> Budgeter:
    """Reset carry to 0 for a fresh budgeting cycle."""
    return Budgeter(per_step=int(b.per_step), carry=0)


__all__.extend(["Budgeter", "admit", "reset"])
