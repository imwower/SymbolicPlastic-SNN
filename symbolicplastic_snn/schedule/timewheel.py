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

        # Enforce strictly increasing and unique indices with fast-path
        if self.indices.size:
            idx = self.indices
            kk = self.k.astype(np.int32, copy=False)
            # Fast path: already strictly increasing -> only clamp k
            if np.all(idx[1:] > idx[:-1]):
                self.k = np.clip(kk, 1, 32767).astype(np.int16, copy=False)
            else:
                order = np.argsort(idx, kind="mergesort")
                idx = idx[order]
                kk = kk[order]
                # Unique and sum duplicates
                uniq, first_idx, counts = np.unique(idx, return_index=True, return_counts=True)
                if uniq.size != idx.size:
                    sums = np.add.reduceat(kk, first_idx)
                    kk = sums
                # Clamp and assign
                kk = np.clip(kk, 1, 32767).astype(np.int16, copy=False)
                self.indices = uniq.astype(np.int32, copy=False)
                self.k = kk


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

    def __init__(self, slots: int, bytes_cap: int | None = None, max_bucket_size: int | None = None, drop_policy: str = "drop_new") -> None:
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
        # Capacity and drop policy for number of groups per slot
        self._max_bucket_size = None if max_bucket_size is None else int(max_bucket_size)
        if drop_policy not in ("drop_oldest", "drop_new", "raise"):
            raise ValueError("drop_policy must be 'drop_oldest'|'drop_new'|'raise'")
        self._drop_policy = drop_policy
        self._seq = 0  # stable insertion sequence per group for drop_oldest

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
            # Enforce per-slot capacity at group creation time
            if self._max_bucket_size is not None and len(bucket) >= int(self._max_bucket_size):
                if self._drop_policy == "drop_new":
                    return
                if self._drop_policy == "raise":
                    raise RuntimeError("TimeWheel bucket capacity exceeded")
                # drop_oldest: remove the group with smallest seq
                if bucket:
                    # find oldest by stored seq
                    oldest_key = None
                    oldest_seq = None
                    for k2, g2 in bucket.items():
                        s2 = int(g2.get("_seq", 0))
                        if oldest_seq is None or s2 < oldest_seq:
                            oldest_seq = s2
                            oldest_key = k2
                    if oldest_key is not None:
                        g_old = bucket.pop(oldest_key)
                        # adjust bytes_used if removing fine-grained
                        if not g_old.get("capped", False):
                            self._bytes_used -= (g_old["idx"].nbytes + g_old["k"].nbytes)
            group = {
                "idx": np.empty((0,), dtype=np.int32),
                "k": np.empty((0,), dtype=np.int16),
                "capped": False,
                "total_k": np.int64(0),
                "_seq": int(self._seq),
            }
            bucket[key] = group
            self._seq += 1

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
        before = self._ptr
        self._ptr = (self._ptr + 1) % self.slots
        # Basic wrap-around sanity: only advance by 1 mod slots
        assert ((before + 1) % self.slots) == self._ptr

    # ---------------- Snapshot/restore ----------------
    def snapshot(self) -> dict:
        """Serialize wheel internal state to a JSON-serializable dict.

        Captures pointer and all pending groups in each slot.
        """
        snap_groups = []
        for s_idx, bucket in enumerate(self._buckets):
            if not bucket:
                continue
            for (post_tile, delay), group in bucket.items():
                g = {
                    "slot": int(s_idx),
                    "post_tile": int(post_tile),
                    "delay": int(delay),
                    "capped": bool(group["capped"]),
                    "total_k": int(group["total_k"]),
                }
                if not group["capped"]:
                    g["idx"] = group["idx"].astype(int).tolist()
                    g["k"] = group["k"].astype(int).tolist()
                else:
                    g["idx"] = []
                    g["k"] = []
                snap_groups.append(g)
        return {
            "slots": int(self.slots),
            "ptr": int(self._ptr),
            "groups": snap_groups,
        }

    def restore(self, snap: dict) -> None:
        """Restore wheel state from a dict created by snapshot()."""
        if not isinstance(snap, dict):
            raise ValueError("Invalid snapshot for TimeWheel")
        slots = int(snap.get("slots", self.slots))
        ptr = int(snap.get("ptr", 0))
        groups = snap.get("groups", []) or []
        if slots != self.slots:
            # Reinitialize buckets with new slot count if mismatched
            self.slots = slots
            self._buckets = [dict() for _ in range(self.slots)]
        else:
            # Clear existing
            self._buckets = [dict() for _ in range(self.slots)]
        self._ptr = ptr % self.slots
        self._bytes_used = 0
        for g in groups:
            s_idx = int(g.get("slot", 0)) % self.slots
            key = (int(g.get("post_tile", 0)), int(g.get("delay", 0)))
            capped = bool(g.get("capped", False))
            total_k = np.int64(int(g.get("total_k", 0)))
            if capped:
                group = {"idx": np.empty((0,), dtype=np.int32), "k": np.empty((0,), dtype=np.int16), "capped": True, "total_k": total_k}
            else:
                idx = np.asarray(g.get("idx", []), dtype=np.int32)
                kk = np.asarray(g.get("k", []), dtype=np.int16)
                group = {"idx": idx, "k": kk, "capped": False, "total_k": total_k}
                self._bytes_used += idx.nbytes + kk.nbytes
            self._buckets[s_idx][key] = group

    # ---------------- Internal helpers ----------------
    def _would_exceed_cap(self, diff_bytes: int) -> bool:
        if self._bytes_cap is None:
            return False
        new_total = self._bytes_used + max(0, int(diff_bytes))
        return new_total > int(self._bytes_cap)

    # ---------------- Convenience ----------------
    def defer_to_future(self, events: Iterable[BlockEvent], extra_delay: int = 1) -> int:
        """Defer events by increasing their delay deterministically and pushing.

        Returns number of events deferred. Arrays are merged per existing policy in push().
        """
        cnt = 0
        d = max(1, int(extra_delay))
        for ev in events:
            be = BlockEvent(
                post_tile=int(ev.post_tile),
                indices=np.array(ev.indices, dtype=np.int32, copy=True),
                k=np.array(ev.k, dtype=np.int16, copy=True),
                delay=int(ev.delay) + d,
                capped=bool(ev.capped),
                total_k=np.int64(ev.total_k),
            )
            self.push(be)
            cnt += 1
        return cnt


# ---- Optional simple event wheel (non-intrusive) ----
from typing import Any


@dataclass
class Event:
    """Basic scheduled event.

    - when: absolute tick within the wheel (0..wheel_size-1)
    - payload: arbitrary object
    - seq: global sequence number for stable ordering within the same tick
    """

    when: int
    payload: Any
    seq: int


class EventWheel:
    """Deterministic cyclic time wheel for Event scheduling.

    API: schedule(delay,payload)->seq; tick()->list[Event] ordered by (when,seq).
    """

    def __init__(self, wheel_size: int) -> None:
        if int(wheel_size) <= 0:
            raise ValueError("wheel_size must be >= 1")
        self.wheel_size = int(wheel_size)
        self.current = 0
        self._seq = 0
        self._buckets: list[list[Event]] = [[] for _ in range(self.wheel_size)]

    def schedule(self, delay: int, payload: Any) -> int:
        if int(delay) < 0:
            raise ValueError("delay must be >= 0")
        seq = self._seq
        self._seq += 1
        when = (self.current + int(delay)) % self.wheel_size
        ev = Event(when=when, payload=payload, seq=seq)
        self._buckets[when].append(ev)
        return seq

    def tick(self) -> list[Event]:
        bucket = self._buckets[self.current]
        out = sorted(bucket, key=lambda e: (int(e.when), int(e.seq))) if bucket else []
        self._buckets[self.current] = []
        self.current = (self.current + 1) % self.wheel_size
        return out

    


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
    # Fast path if both are strictly increasing (most common): linear merge
    def _is_incr(x: NDArray[np.int32]) -> bool:
        return x.size == 0 or bool(np.all(x[1:] > x[:-1]))

    if _is_incr(a_idx) and _is_incr(b_idx):
        a_n = a_idx.size
        b_n = b_idx.size
        if a_n == 0:
            return b_idx.astype(np.int32, copy=False), np.clip(b_k, 1, 32767).astype(np.int16, copy=False)
        if b_n == 0:
            return a_idx.astype(np.int32, copy=False), np.clip(a_k, 1, 32767).astype(np.int16, copy=False)
        out_idx = np.empty(a_n + b_n, dtype=np.int32)
        out_k = np.empty(a_n + b_n, dtype=np.int32)
        ai = bi = oi = 0
        a_ki = a_k.astype(np.int32, copy=False)
        b_ki = b_k.astype(np.int32, copy=False)
        while ai < a_n and bi < b_n:
            av = int(a_idx[ai])
            bv = int(b_idx[bi])
            if av < bv:
                out_idx[oi] = av
                out_k[oi] = a_ki[ai]
                ai += 1
                oi += 1
            elif av > bv:
                out_idx[oi] = bv
                out_k[oi] = b_ki[bi]
                bi += 1
                oi += 1
            else:
                out_idx[oi] = av
                out_k[oi] = a_ki[ai] + b_ki[bi]
                ai += 1
                bi += 1
                oi += 1
        # Append remainders
        if ai < a_n:
            n = a_n - ai
            out_idx[oi:oi+n] = a_idx[ai:]
            out_k[oi:oi+n] = a_ki[ai:]
            oi += n
        if bi < b_n:
            n = b_n - bi
            out_idx[oi:oi+n] = b_idx[bi:]
            out_k[oi:oi+n] = b_ki[bi:]
            oi += n
        out_idx = out_idx[:oi]
        out_k = out_k[:oi]
        out_k = np.clip(out_k, 1, 32767).astype(np.int16, copy=False)
        return out_idx, out_k

    # Fallback: concatenate, sort, unique + reduceat
    if a_idx.size == 0:
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
    uniq, first_idx, _ = np.unique(idx, return_index=True, return_counts=True)
    sums = np.add.reduceat(kk, first_idx)
    sums = np.clip(sums, 1, 32767).astype(np.int16, copy=False)
    return uniq.astype(np.int32, copy=False), sums
