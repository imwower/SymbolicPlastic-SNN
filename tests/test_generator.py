import numpy as np

from symbolicplastic_snn.conn.alias import AliasForTile, build_alias
from symbolicplastic_snn.conn.generator import gen_block_events, JITTER_MAX
from symbolicplastic_snn.realtime.budgeter import EventBudget


def make_uniform_alias(ntiles: int) -> AliasForTile:
    base = 65535 // ntiles
    w = np.full(ntiles, base, dtype=np.uint16)
    rem = 65535 - base * ntiles
    if rem > 0:
        w[:rem] = (w[:rem].astype(np.uint32) + 1).astype(np.uint16)
    prob, alias = build_alias(w)
    return AliasForTile(prob=prob, alias=alias)


def test_quota_aggregate():
    ntiles = 8
    alias_tbl = make_uniform_alias(ntiles)
    tile_size = 64
    M = 16
    T_tiles = 40  # expect duplicates across 8 tiles
    seed = np.uint64(123456789)

    events = gen_block_events(
        pre_id=7,
        step=10,
        pre_tile=3,
        alias_tbl=alias_tbl,
        tile_size=tile_size,
        T_tiles=T_tiles,
        M=M,
        seed=seed,
        delay_lut=np.zeros((ntiles, ntiles), dtype=np.uint8),
        budget=None,
    )

    # No duplicate post_tile entries; duplicates must have been aggregated
    post_tiles = [ev.post_tile for ev in events]
    assert len(post_tiles) == len(set(post_tiles))
    # Each event carries exactly M indices
    assert all(ev.indices.size == M for ev in events)


def test_budget_respected():
    ntiles = 6
    alias_tbl = make_uniform_alias(ntiles)
    tile_size = 64
    M = 16
    T_tiles = 50
    seed = np.uint64(987654321)

    budget = EventBudget(b_step=M)  # allow only one event worth of cost
    events = gen_block_events(
        pre_id=1,
        step=2,
        pre_tile=0,
        alias_tbl=alias_tbl,
        tile_size=tile_size,
        T_tiles=T_tiles,
        M=M,
        seed=seed,
        delay_lut=np.zeros((ntiles, ntiles), dtype=np.uint8),
        budget=budget,
    )

    assert len(events) == 1
    assert events[0].indices.size == M
    assert budget.used == M


def test_determinism():
    ntiles = 5
    alias_tbl = make_uniform_alias(ntiles)
    tile_size = 32
    M = 8
    T_tiles = 20
    seed1 = np.uint64(555)
    seed2 = np.uint64(556)

    args = dict(
        pre_id=9,
        step=42,
        pre_tile=4,
        alias_tbl=alias_tbl,
        tile_size=tile_size,
        T_tiles=T_tiles,
        M=M,
        delay_lut=np.ones((ntiles, ntiles), dtype=np.uint8),
    )

    ev1 = gen_block_events(seed=seed1, budget=None, **args)
    ev2 = gen_block_events(seed=seed1, budget=None, **args)
    ev3 = gen_block_events(seed=seed2, budget=None, **args)

    # Same seed -> identical event lists
    assert len(ev1) == len(ev2)
    for a, b in zip(ev1, ev2):
        assert a.post_tile == b.post_tile
        assert a.delay == b.delay
        assert np.array_equal(a.indices, b.indices)
        assert np.array_equal(a.k, b.k)

    # Different seed -> very likely different in tiles or delays
    different = (
        len(ev1) != len(ev3)
        or any(a.post_tile != b.post_tile for a, b in zip(ev1, ev3))
        or any(a.delay != b.delay for a, b in zip(ev1, ev3))
    )
    assert different


def test_delay_lut():
    ntiles = 4
    alias_tbl = make_uniform_alias(ntiles)
    tile_size = 16
    M = 8
    T_tiles = 10
    seed = np.uint64(999)

    # Construct simple LUT: base = pre_tile + post_tile
    base = np.fromfunction(lambda i, j: i + j, (ntiles, ntiles), dtype=int).astype(np.uint8)

    events = gen_block_events(
        pre_id=2,
        step=3,
        pre_tile=1,
        alias_tbl=alias_tbl,
        tile_size=tile_size,
        T_tiles=T_tiles,
        M=M,
        seed=seed,
        delay_lut=base,
        budget=None,
    )

    for ev in events:
        b = int(base[1, ev.post_tile])
        assert b <= ev.delay <= b + JITTER_MAX

