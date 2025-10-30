from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class BlockEvent:
    """Aggregated event for one post tile and delay.

    - indices: strictly ascending and unique int32 indices
    - k: int16 strengths aligned to indices, each k >= 1
    - delay: non-negative integer delay used for grouping and scheduling

    When memory cap forces coarse aggregation, `capped` is True and
    indices/k will be empty arrays while `total_k` carries the total strength.
    """

    post_tile: int
    indices: NDArray[np.int32]
    k: NDArray[np.int16]
    delay: int

    # Coarse aggregation flag and total strength for tile-level aggregation.
    capped: bool = False
    total_k: np.int64 | int = np.int64(0)

    def __post_init__(self) -> None:
        # Normalize dtypes
        if self.indices.dtype != np.int32:
            self.indices = self.indices.astype(np.int32, copy=False)
        if self.k.dtype != np.int16:
            self.k = self.k.astype(np.int16, copy=False)

        # Ensure arrays are 1-D
        self.indices = np.ravel(self.indices)
        self.k = np.ravel(self.k)

        if self.indices.size != self.k.size:
            raise ValueError("indices and k must have the same length")
        if self.delay < 0:
            raise ValueError("delay must be non-negative")

        # Enforce strictly increasing and unique indices, stable deterministic
        if self.indices.size:
            order = np.argsort(self.indices, kind="mergesort")
            idx = self.indices[order]
            kk = self.k.astype(np.int32, copy=False)[order]

            # Unique and sum duplicates
            uniq, first_idx, counts = np.unique(idx, return_index=True, return_counts=True)
            if uniq.size != idx.size:  # there are duplicates
                sums = np.add.reduceat(kk, first_idx)
                # Add contributions of following duplicates (counts>1)
                dup_mask = counts > 1
                if np.any(dup_mask):
                    # For duplicates, reduceat already sums contiguous groups
                    pass
                kk_new = sums
            else:
                kk_new = kk

            # Clamp k to int16 range [1, 32767] and cast to int16
            kk_new = np.clip(kk_new, 1, 32767).astype(np.int16, copy=False)
            self.indices = uniq.astype(np.int32, copy=False)
            self.k = kk_new


class TimeWheel:
    """A deterministic time wheel with per-(tile, delay) aggregation and memory cap.

    - Events are scheduled to slot = (pointer + delay) % slots
    - Within each slot, events with the same (post_tile, delay) are merged:
      same indices aggregate by summing k. indices remain ascending and unique.
    - If total memory exceeds `bytes_cap`, the offending group is downgraded to
      a coarse aggregated representation (tile-level). Such events set `capped=True`
      and expose `total_k` while `indices`/`k` are empty arrays.
    - All operations are stable and deterministic.
    """

    def __init__(self, slots: int, bytes_cap: int | None = None) -> None:
        if slots <= 0:
            raise ValueError("slots must be positive")
        self.slots = int(slots)
        self._ptr = 0
        # Each slot holds a dict keyed by (post_tile, delay)
        # value: dict with keys: idx (int32 ndarray), k (int16 ndarray),
        #        capped (bool), total_k (int64)
        self._buckets: List[Dict[Tuple[int, int], dict]] = [dict() for _ in range(self.slots)]
        self._bytes_cap = bytes_cap if bytes_cap is None else int(bytes_cap)
        self._bytes_used = 0  # only counts ndarray buffer bytes of indices/k

    # ---------------- Public API ----------------
    def push(self, event: BlockEvent) -> None:
        """Push one BlockEvent into the wheel with merge policy.

        If memory cap would be exceeded by the finer aggregation, downgrade the
        target group to coarse aggregation (tile-level) deterministically.
        """
        ev = normalize_block_event(event)
        key = (int(ev.post_tile), int(ev.delay))
        slot_idx = (self._ptr + int(ev.delay)) % self.slots
        bucket = self._buckets[slot_idx]

        group = bucket.get(key)
        if group is None:
            group = {
                "idx": np.empty((0,), dtype=np.int32),
                "k": np.empty((0,), dtype=np.int16),
                "capped": False,
                "total_k": np.int64(0),
            }
            bucket[key] = group

        if group["capped"]:
            # Already coarse; accumulate total strength only
            group["total_k"] = np.int64(group["total_k"]) + np.int64(np.int64(ev.k.astype(np.int64).sum()))
            return

        # Merge arrays deterministically
        new_idx, new_k = _merge_indices_k(group["idx"], group["k"], ev.indices, ev.k)
        # Calculate memory diff
        prev_bytes = group["idx"].nbytes + group["k"].nbytes
        new_bytes = new_idx.nbytes + new_k.nbytes
        diff = new_bytes - prev_bytes

        # Check cap: if exceeded, downgrade to coarse aggregation
        if self._would_exceed_cap(diff):
            # Remove previous arrays accounting from bytes_used
            self._bytes_used -= prev_bytes
            # Coarse aggregate total_k = previous sums + new sums
            total_prev = np.int64(group["k"].astype(np.int64).sum()) if group["k"].size else np.int64(0)
            total_new = np.int64(ev.k.astype(np.int64).sum())
            group["idx"] = np.empty((0,), dtype=np.int32)
            group["k"] = np.empty((0,), dtype=np.int16)
            group["capped"] = True
            group["total_k"] = np.int64(group["total_k"]) + total_prev + total_new
            # Capped groups do not contribute array buffer bytes
            return

        # Commit fine-grained arrays
        group["idx"] = new_idx
        group["k"] = new_k
        self._bytes_used += diff

    def push_batch(self, events: Iterable[BlockEvent]) -> None:
        for ev in events:
            self.push(ev)

    def pop(self) -> List[BlockEvent]:
        """Pop and clear current slot, returning aggregated events list.

        Output order is deterministic: sorted by (post_tile, delay).
        """
        bucket = self._buckets[self._ptr]
        if not bucket:
            return []

        # Prepare deterministic order
        keys = sorted(bucket.keys())  # (post_tile, delay)
        out: List[BlockEvent] = []

        for key in keys:
            post_tile, delay = key
            group = bucket[key]
            if group["capped"]:
                be = BlockEvent(
                    post_tile=post_tile,
                    indices=np.empty((0,), dtype=np.int32),
                    k=np.empty((0,), dtype=np.int16),
                    delay=delay,
                    capped=True,
                    total_k=np.int64(group["total_k"]),
                )
            else:
                be = BlockEvent(
                    post_tile=post_tile,
                    indices=group["idx"],
                    k=group["k"],
                    delay=delay,
                    capped=False,
                    total_k=np.int64(0),
                )
                # Deduct bytes for fine-grained arrays
                self._bytes_used -= (group["idx"].nbytes + group["k"].nbytes)

            out.append(be)

        # Clear the bucket
        self._buckets[self._ptr] = {}

        return out

    def tick(self) -> None:
        self._ptr = (self._ptr + 1) % self.slots

    # ---------------- Internal helpers ----------------
    def _would_exceed_cap(self, diff_bytes: int) -> bool:
        if self._bytes_cap is None:
            return False
        new_total = self._bytes_used + max(0, int(diff_bytes))
        return new_total > int(self._bytes_cap)


def normalize_block_event(event: BlockEvent) -> BlockEvent:
    """Return a normalized copy of BlockEvent with constraints enforced.

    - indices ascending and unique
    - k aligned, int16, >= 1
    - delay >= 0
    """
    # Construct a new BlockEvent to trigger __post_init__ checks and sorting
    # Also coerce k to be >= 1 (clip)
    idx = np.asarray(event.indices, dtype=np.int32)
    kk = np.asarray(event.k, dtype=np.int16)
    # Ensure k >= 1
    kk = np.clip(kk.astype(np.int32), 1, 32767).astype(np.int16)

    return BlockEvent(
        post_tile=int(event.post_tile),
        indices=idx,
        k=kk,
        delay=int(event.delay),
        capped=bool(event.capped),
        total_k=np.int64(event.total_k),
    )


def _merge_indices_k(
    a_idx: NDArray[np.int32],
    a_k: NDArray[np.int16],
    b_idx: NDArray[np.int32],
    b_k: NDArray[np.int16],
) -> Tuple[NDArray[np.int32], NDArray[np.int16]]:
    """Deterministically merge two (indices, k) arrays.

    - Result indices strictly ascending and unique
    - For duplicate indices, k are summed
    - k is saturated into int16 range [1, 32767]
    """
    if a_idx.size == 0:
        # Ensure normalization on the input
        if b_idx.size == 0:
            return a_idx, a_k
        order = np.argsort(b_idx, kind="mergesort")
        idx = b_idx[order]
        kk = b_k.astype(np.int32, copy=False)[order]
    elif b_idx.size == 0:
        order = np.argsort(a_idx, kind="mergesort")
        idx = a_idx[order]
        kk = a_k.astype(np.int32, copy=False)[order]
    else:
        idx = np.concatenate((a_idx, b_idx))
        kk = np.concatenate((a_k.astype(np.int32), b_k.astype(np.int32)))
        order = np.argsort(idx, kind="mergesort")
        idx = idx[order]
        kk = kk[order]

    # Unique indices and sum duplicates deterministically
    uniq, first_idx, counts = np.unique(idx, return_index=True, return_counts=True)
    sums = np.add.reduceat(kk, first_idx)
    # Saturate to int16 and ensure k >= 1
    sums = np.clip(sums, 1, 32767).astype(np.int16, copy=False)
    return uniq.astype(np.int32, copy=False), sums

