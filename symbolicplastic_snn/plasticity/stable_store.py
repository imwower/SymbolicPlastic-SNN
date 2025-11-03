from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

from .states import EDGE_STABLE


@dataclass
class StableEdge:
    pre_id: int
    post_id: int
    sign: np.int8  # ±1
    delay: np.uint8
    state: np.uint8 = np.uint8(EDGE_STABLE)
    age: np.uint16 = np.uint16(0)
    corr: np.int16 = np.int16(0)
    last_used: np.int32 = np.int32(0)


class StableStore:
    """In-memory store for per-pre pinned connections.

    - Deterministic insertion order preserved per pre_id
    - Capacity enforced per pre: at most `per_pre_cap` entries
    """

    def __init__(self, per_pre_cap: int = 128) -> None:
        self.per_pre_cap = int(per_pre_cap)
        # pre_id -> Ordered map: (post_id, delay) -> StableEdge
        self._by_pre: Dict[int, Dict[Tuple[int, int], StableEdge]] = {}

    def get_pre(self, pre_id: int) -> List[StableEdge]:
        m = self._by_pre.get(int(pre_id))
        if not m:
            return []
        # Return in insertion order
        return list(m.values())

    def add(self, edge: StableEdge) -> bool:
        pid = int(edge.pre_id)
        key = (int(edge.post_id), int(edge.delay))
        m = self._by_pre.get(pid)
        if m is None:
            m = {}
            self._by_pre[pid] = m
        if key in m:
            # Already exists; do not duplicate
            return False
        if len(m) >= self.per_pre_cap:
            return False
        m[key] = edge
        return True

    def remove(self, pre_id: int, post_id: int, delay: int) -> bool:
        pid = int(pre_id)
        key = (int(post_id), int(delay))
        m = self._by_pre.get(pid)
        if not m or key not in m:
            return False
        del m[key]
        if not m:
            del self._by_pre[pid]
        return True

    def exists(self, pre_id: int, post_id: int, delay: int) -> bool:
        pid = int(pre_id)
        key = (int(post_id), int(delay))
        m = self._by_pre.get(pid)
        return bool(m and key in m)

    def iter_all(self) -> Iterator[StableEdge]:
        for pid in sorted(self._by_pre.keys()):
            for e in self._by_pre[pid].values():
                yield e

    def stats(self) -> Dict[str, int]:
        counts = [len(m) for m in self._by_pre.values()]
        total = sum(counts)
        return {
            "per_pre_cap": int(self.per_pre_cap),
            "num_pre": len(counts),
            "total_edges": total,
            "max_per_pre": max(counts) if counts else 0,
        }


__all__ = ["StableEdge", "StableStore"]

