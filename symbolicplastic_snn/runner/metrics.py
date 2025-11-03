from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

import numpy as np

from symbolicplastic_snn.plasticity.states import EDGE_CONS


@dataclass
class MetricsTracker:
    promoted_count: int = 0
    demoted_count: int = 0
    sign_flip_count: int = 0
    frozen_count: int = 0
    stable_edges_total: int = 0
    per_pre_mean: float = 0.0
    per_pre_std: float = 0.0
    used_budget_ratio: float = 0.0
    deferred_events: int = 0

    def update_from_runner(self, runner, pipeline_out: Optional[Dict[str, int | np.ndarray]] = None) -> None:
        # Incremental counters from pipeline (if provided)
        if pipeline_out is not None:
            self.promoted_count += int(pipeline_out.get("promoted", 0))
            self.demoted_count += int(pipeline_out.get("demoted", 0))
            self.sign_flip_count += int(pipeline_out.get("flipped", 0))

        # Stable store stats
        store = getattr(runner, "stable_store", None)
        per_counts: Dict[int, int] = {}
        frozen = 0
        total = 0
        if store is not None:
            for e in store.iter_all():
                total += 1
                per_counts[int(e.pre_id)] = per_counts.get(int(e.pre_id), 0) + 1
                if int(e.state) == EDGE_CONS:
                    frozen += 1
        self.stable_edges_total = int(total)
        self.frozen_count = int(frozen)
        if per_counts:
            arr = np.array(list(per_counts.values()), dtype=np.float64)
            self.per_pre_mean = float(arr.mean())
            self.per_pre_std = float(arr.std(ddof=0))
        else:
            self.per_pre_mean = 0.0
            self.per_pre_std = 0.0

        # Runner realtime metrics
        lm = getattr(runner, "last_metrics", {}) or {}
        # prefer used_budget_ratio if present, else budget_used_ratio
        ubr = lm.get("used_budget_ratio", lm.get("budget_used_ratio", 0.0))
        self.used_budget_ratio = float(ubr)
        self.deferred_events = int(lm.get("deferred_events", lm.get("deferred_count", 0)))

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        # Clamp floats to sane range
        d["used_budget_ratio"] = float(max(0.0, min(1.0, self.used_budget_ratio)))
        return d

    def reset(self) -> None:
        self.promoted_count = 0
        self.demoted_count = 0
        self.sign_flip_count = 0
        self.frozen_count = 0
        self.stable_edges_total = 0
        self.per_pre_mean = 0.0
        self.per_pre_std = 0.0
        self.used_budget_ratio = 0.0
        self.deferred_events = 0

    @staticmethod
    def should_report(step: int, period: int = 500) -> bool:
        if period <= 0:
            return True
        return (int(step) % int(period)) == 0


__all__ = ["MetricsTracker"]

