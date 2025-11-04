from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from symbolicplastic_snn.conn.alias import AliasForTile, sample_alias
from symbolicplastic_snn.conn.permute import permute_first_m
from symbolicplastic_snn.realtime.budgeter import EventBudget
from symbolicplastic_snn.schedule.timewheel import BlockEvent


JITTER_MAX = 3  # inclusive max added to LUT base delay
_MASK64 = (1 << 64) - 1


def _xs64star_next(state: int) -> tuple[int, int]:
    x = state & _MASK64
    x ^= (x >> 12) & _MASK64
    x ^= ((x << 25) & _MASK64)
    x ^= (x >> 27) & _MASK64
    new_state = x & _MASK64
    z = (new_state * 2685821657736338717) & _MASK64
    return new_state, z


def _rng_uint16_from_seed(seed: int):
    state = int(seed) & _MASK64

    def rng_with_state(size: int) -> np.ndarray:
        nonlocal state
        out = np.empty(size, dtype=np.uint16)
        s = state
        for i in range(size):
            s, r = _xs64star_next(s)
            out[i] = np.uint16((r >> 48) & 0xFFFF)
        state = s
        return out

    return rng_with_state


def _mix_key(pre_id: int, post_tile: int, step: int) -> int:
    """Derive a deterministic 64-bit key using xorshift64* mixing."""
    x = (int(pre_id) * 0xD1342543DE82EF95) & _MASK64
    x ^= ((int(post_tile) + 0x9E37) * 0xC2B2AE3D27D4EB4F) & _MASK64
    x ^= (int(step) * 0x165667B19E3779F9) & _MASK64
    # One xorshift64* round for diffusion
    _, z = _xs64star_next(x)
    return z


def gen_block_events(
    pre_id: int,
    step: int,
    pre_tile: int,
    alias_for_tile,
    tile_size: int,
    T_tiles: int = 32,
    M: int = 128,
    seed: np.uint64 | int | None = None,
    delay_lut: np.ndarray | None = None,
    budget: EventBudget | None = None,
    *,
    # New API parameters (preferred):
    seed_core: np.uint64 | int | None = None,
    seed_flex: np.uint64 | int | None = None,
    core_ratio: float = 0.8,
    rng: callable | None = None,
    stable_store: object | None = None,
) -> List[BlockEvent]:
    """Programmatically generate connectivity events for one pre tile.

    Supports legacy and new APIs:
    - Legacy: provide `alias_for_tile` as AliasForTile, `seed` (single), optional `budget`.
    - New: provide `alias_for_tile` as (prob:uint16[], alias:int32[]), `seed_core`,
      `seed_flex`, `core_ratio` (0..1), and `rng` for jitter (rng(size)->uint16 array).

    Steps
    1) Sample T_tiles post tiles with replacement via core/flex channels per core_ratio.
       Aggregate quotas per tile by summing counts.
    2) For each post tile: key = mix(pre_id, post_tile, step) → permute_first_m.
    3) delay = delay_lut[pre_tile, post_tile] + jitter(rng) in [0, JITTER_MAX]; clip to uint8.
    4) Emit BlockEvent with sorted unique indices and uniform k = quota (>=1).

    Deterministic: all randomness derived from seeds and rng.
    """
    if tile_size <= 0:
        return []
    if M <= 0:
        return []
    if M > tile_size:
        raise ValueError("M must be <= tile_size")
    if T_tiles <= 0:
        return []

    # Unpack alias inputs
    if isinstance(alias_for_tile, AliasForTile):
        prob = alias_for_tile.prob
        alias_idx = alias_for_tile.alias
    else:
        prob, alias_idx = alias_for_tile
        prob = np.asarray(prob, dtype=np.uint16)
        alias_idx = np.asarray(alias_idx, dtype=np.int32)

    # RNGs for alias sampling: derive from provided seeds
    if seed_core is None and seed_flex is None:
        # Legacy path: derive from single seed if provided, else 0
        seed_val = 0 if seed is None else int(np.uint64(seed))
        seed_core = (seed_val ^ 0xD1342543DE82EF95) & _MASK64
        seed_flex = (seed_val ^ 0x94D049BB133111EB) & _MASK64
    core_n = int(max(0, min(T_tiles, int(T_tiles))))
    # Determine split deterministically
    c_n = int(max(0, min(T_tiles, int(float(core_ratio) * int(T_tiles)))))
    f_n = int(T_tiles) - c_n
    rng16_core = _rng_uint16_from_seed(int(np.uint64(seed_core)))
    rng16_flex = _rng_uint16_from_seed(int(np.uint64(seed_flex)))

    # 0) Inject stable connections (group by (post_tile, delay), sum k per local index)
    stable_groups: dict[tuple[int, int], dict[int, int]] = {}
    included_stable: set[tuple[int, int]] = set()
    if stable_store is not None and hasattr(stable_store, "get_pre"):
        try:
            edges = list(stable_store.get_pre(int(pre_id)))
        except Exception:
            edges = []
        for e in edges:
            post_id = int(getattr(e, "post_id"))
            delay_e = int(getattr(e, "delay"))
            post_tile_e = post_id // int(tile_size)
            local_idx = post_id % int(tile_size)
            key = (post_tile_e, delay_e)
            g = stable_groups.get(key)
            if g is None:
                g = {}
                stable_groups[key] = g
            g[local_idx] = g.get(local_idx, 0) + 1
        # Apply budget for stable groups first (priority)
        for key, g in stable_groups.items():
            cnt = len(g)
            if budget is not None and not budget.allow(cnt):
                continue
            if budget is not None:
                budget.charge(cnt)
            included_stable.add(key)

    # 1) Sample post tiles via core and flex channels, then aggregate
    tiles_core = sample_alias(prob, alias_idx, rng16_core, int(c_n)) if c_n > 0 else np.zeros(0, dtype=np.int32)
    tiles_flex = sample_alias(prob, alias_idx, rng16_flex, int(f_n)) if f_n > 0 else np.zeros(0, dtype=np.int32)
    tiles = np.concatenate((tiles_core, tiles_flex)) if (c_n + f_n) > 0 else np.zeros(0, dtype=np.int32)
    quota = Counter(tiles.tolist())

    # Deterministic order over unique post tiles
    unique_tiles = sorted(quota.keys())

    events: List[BlockEvent] = []

    # base delays from LUT if provided
    use_lut = delay_lut is not None
    if use_lut:
        lut = np.asarray(delay_lut)
        if lut.ndim != 2:
            raise ValueError("delay_lut must be 2-D [pre_tile, post_tile]")

    # For per-tile jitter, derive from RNG sequence by drawing one 64-bit value
    # via composing four 16-bit outputs for determinism.
    # Jitter RNG uses provided rng if available, else derive from core seed
    if rng is None:
        rng = _rng_uint16_from_seed(int(np.uint64(seed_core)))

    def next_jitter() -> int:
        r = int(np.asarray(rng(1), dtype=np.uint16)[0])
        return int(r % (JITTER_MAX + 1))

    for post_tile in unique_tiles:
        q = int(quota[post_tile])
        key = _mix_key(pre_id, int(post_tile), step)
        indices = permute_first_m(int(tile_size), int(M), key)
        # Sort indices ascending to help downstream fast-path merging
        indices = np.sort(indices, kind="mergesort")

        base_delay = 0
        if use_lut:
            base_delay = int(np.asarray(lut[pre_tile, post_tile]).item())
        jitter = next_jitter()
        delay = base_delay + jitter
        if delay < 0:
            delay = 0
        if delay > 255:
            delay = 255
        key2 = (int(post_tile), int(delay))
        # Merge with stable group if present (avoid duplicate indices). Overlaps -> k += 1
        if key2 in included_stable:
            g = stable_groups[key2]
            # Determine indices to add from variable selection respecting budget
            added = []
            for idx in indices.tolist():
                if idx in g:
                    # variable quota contributes q more connections to this index
                    g[idx] = int(g[idx]) + int(q)
                else:
                    added.append(int(idx))
            # Apply budget for new indices only
            add_cnt = len(added)
            if add_cnt > 0:
                if budget is None or budget.allow(add_cnt):
                    if budget is not None:
                        budget.charge(add_cnt)
                    for v in added:
                        g[v] = int(q)  # variable quota contributes q for new entries
                else:
                    # Add as much as budget allows deterministically
                    can = budget.remaining if budget is not None else 0
                    if can > 0:
                        take = added[:can]
                        for v in take:
                            g[v] = int(q)
                        if budget is not None:
                            budget.charge(len(take))
            # No separate event appended; this group will be finalized later
        else:
            # No stable group; create a fresh event if budget allows
            cost = int(indices.size)
            if budget is not None and not budget.allow(cost):
                continue
            if budget is not None:
                budget.charge(cost)
            k = np.full(indices.shape[0], q, dtype=np.int16)
            events.append(
                BlockEvent(
                    post_tile=int(post_tile),
                    indices=indices,
                    k=k,
                    delay=int(delay),
                )
            )

    # Finalize included stable groups as BlockEvents
    for (pt, dly) in included_stable:
        g = stable_groups[(pt, dly)]
        if not g:
            continue
        idx_sorted = np.array(sorted(g.keys()), dtype=np.int32)
        kk = np.array([int(g[i]) for i in idx_sorted.tolist()], dtype=np.int16)
        events.append(
            BlockEvent(
                post_tile=int(pt),
                indices=idx_sorted,
                k=kk,
                delay=int(dly),
            )
        )

    return events


__all__ = ["gen_block_events", "JITTER_MAX"]
 
 
def gen_block_events_batch(
    pre_ids: Sequence[int],
    step: int,
    pre_tiles: Sequence[int],
    alias_tbls: Sequence[AliasForTile],
    tile_size: int,
    T_tiles: int,
    M: int,
    seeds: Sequence[int | np.uint64],
    delay_lut: np.ndarray | None = None,
    *,
    stable_store: object | None = None,
    rng_list: Sequence[callable] | None = None,
) -> List[Tuple[BlockEvent, int]]:
    """Batch wrapper that generates events for multiple presynaptic neurons.

    Returns a list of (BlockEvent, pre_tile) pairs in the same order as inputs,
    preserving determinism with per-pre seeds.
    """
    n = len(pre_ids)
    if not (len(pre_tiles) == n and len(alias_tbls) == n and len(seeds) == n):
        raise ValueError("pre_ids, pre_tiles, alias_tbls, seeds must have same length")
    out: List[Tuple[BlockEvent, int]] = []
    for i in range(n):
        pid = int(pre_ids[i])
        ptile = int(pre_tiles[i])
        alias = alias_tbls[i]
        seed = seeds[i]
        evs = gen_block_events(
            pre_id=pid,
            step=step,
            pre_tile=ptile,
            alias_for_tile=alias,
            tile_size=int(tile_size),
            T_tiles=int(T_tiles),
            M=int(M),
            seed=np.uint64(seed),
            delay_lut=delay_lut,
            budget=None,
            stable_store=stable_store,
            rng=(rng_list[i] if rng_list is not None else None),
        )
        for ev in evs:
            out.append((ev, ptile))
    return out

__all__.append("gen_block_events_batch")
