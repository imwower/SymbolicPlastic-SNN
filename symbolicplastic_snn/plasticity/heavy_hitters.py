from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


class SpaceSavingK:
    """Space-Saving heavy-hitter tracker with fixed capacity.

    Uses three parallel arrays (post_id, count, err) of length `capacity`.
    - update(post_id):
        If tracked -> count++
        Else if free slot -> insert (id, 1, 0)
        Else -> replace minimal count slot with (id, cmin+1, cmin)
    - topk(): returns list of (post_id, count, err) sorted by count desc.
    - clear(): removes all tracked items.

    Note: counts are approximate upper bounds; error is at most the count at
    replacement time (per Space-Saving algorithm).
    """

    def __init__(self, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._cap = int(capacity)
        self._ids = np.full(self._cap, -1, dtype=np.int32)
        self._cnt = np.zeros(self._cap, dtype=np.int32)
        self._err = np.zeros(self._cap, dtype=np.int32)
        self._size = 0  # number of used slots

    @property
    def capacity(self) -> int:
        return self._cap

    def update(self, post_id: int) -> None:
        pid = int(post_id)
        # Try find existing id
        if self._size > 0:
            # Linear scan (capacity is small); keeps memory minimal
            for i in range(self._size):
                if int(self._ids[i]) == pid:
                    self._cnt[i] = int(self._cnt[i]) + 1
                    return
        # Insert if free slot available
        if self._size < self._cap:
            i = self._size
            self._ids[i] = np.int32(pid)
            self._cnt[i] = np.int32(1)
            self._err[i] = np.int32(0)
            self._size += 1
            return
        # Replace minimal count slot
        # Find index with minimal count; stable tie-breaking by lowest index
        i_min = 0
        c_min = int(self._cnt[0])
        for i in range(1, self._cap):
            ci = int(self._cnt[i])
            # Tie-break by picking the latter index to avoid evicting long-lived items
            if ci <= c_min:
                c_min = ci
                i_min = i
        self._ids[i_min] = np.int32(pid)
        # Per algorithm, new count = c_min + 1, error = c_min
        self._err[i_min] = np.int32(c_min)
        self._cnt[i_min] = np.int32(c_min + 1)

    def topk(self) -> List[Tuple[int, int, int]]:
        if self._size == 0:
            return []
        ids = self._ids[: self._size]
        cnt = self._cnt[: self._size]
        err = self._err[: self._size]
        order = np.argsort(-cnt.astype(np.int64), kind="mergesort")
        out: List[Tuple[int, int, int]] = []
        for i in order:
            out.append((int(ids[i]), int(cnt[i]), int(err[i])))
        return out

    def clear(self) -> None:
        self._ids.fill(-1)
        self._cnt.fill(0)
        self._err.fill(0)
        self._size = 0


__all__ = ["SpaceSavingK"]
