from __future__ import annotations

from collections import Counter
from typing import List

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
    alias_tbl: AliasForTile,
    tile_size: int,
    T_tiles: int = 32,
    M: int = 128,
    seed: np.uint64 | int = 0,
    delay_lut: np.ndarray | None = None,
    budget: EventBudget | None = None,
) -> List[BlockEvent]:
    """Programmatically generate connectivity events for one pre tile.

    Steps
    1) Sample T_tiles post tiles with replacement, aggregate quotas per tile.
    2) For each post tile: key = mix(pre_id, post_tile, step) → permute_first_m.
    3) delay = delay_lut[pre_tile, post_tile] + jitter (0..JITTER_MAX) from seed.
    4) Emit BlockEvent with indices and uniform k = quota, respecting optional budget.

    Deterministic: all randomness derived from fixed SplitMix64 sequences.
    """
    if tile_size <= 0:
        return []
    if M <= 0:
        return []
    if M > tile_size:
        raise ValueError("M must be <= tile_size")
    if T_tiles <= 0:
        return []

    # RNG for alias sampling and jitter, derived from seed + identifiers.
    seed_val = int(np.uint64(seed))
    seed_mix = (seed_val ^ (pre_id * 0x9E3779B1) ^ (pre_tile * 0xC2B2AE35) ^ (step * 0x165667B1)) & _MASK64
    rng16 = _rng_uint16_from_seed(seed_mix)

    # 1) Sample post tiles
    tiles = sample_alias(alias_tbl.prob, alias_tbl.alias, rng16, int(T_tiles))
    # Aggregate quotas per tile
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
    def next_jitter() -> int:
        # Compose jitter from next PRN
        r = int(rng16(1)[0])
        # Expand to small jitter 0..JITTER_MAX using upper bits
        return int((r >> 13) & JITTER_MAX)

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

        # Event cost
        cost = int(indices.size)
        if budget is not None:
            if not budget.allow(cost):
                break
            budget.charge(cost)

        # Strength vector: each chosen index gets k = quota
        k = np.full(indices.shape[0], q, dtype=np.int16)
        events.append(
            BlockEvent(
                post_tile=int(post_tile),
                indices=indices,
                k=k,
                delay=int(delay),
            )
        )

    return events


__all__ = ["gen_block_events", "JITTER_MAX"]
