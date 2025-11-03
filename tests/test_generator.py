import numpy as np

from symbolicplastic_snn.conn.alias import build_alias, sample_alias
from symbolicplastic_snn.conn.generator import gen_block_events, JITTER_MAX


def _xs64star_rng_uint16(seed: int):
    MASK = (1 << 64) - 1
    MUL = 2685821657736338717
    state = int(seed) & MASK

    def next_u64():
        nonlocal state
        x = state & MASK
        x ^= (x >> 12) & MASK
        x ^= ((x << 25) & MASK)
        x ^= (x >> 27) & MASK
        state = x & MASK
        return (state * MUL) & MASK

    def rng(size: int) -> np.ndarray:
        out = np.empty(size, dtype=np.uint16)
        for i in range(size):
            out[i] = np.uint16((next_u64() >> 48) & 0xFFFF)
        return out

    return rng


def _uniform_alias(ntiles: int):
    base = 65535 // ntiles
    w = np.full(ntiles, base, dtype=np.uint16)
    rem = 65535 - base * ntiles
    if rem > 0:
        w[:rem] = (w[:rem].astype(np.uint32) + 1).astype(np.uint16)
    return build_alias(w)


def test_quota_aggregation_by_tile():
    ntiles = 8
    prob, alias = _uniform_alias(ntiles)
    tile_size = 64
    M = 16
    T_tiles = 40
    seed_core = np.uint64(123456789)
    seed_flex = np.uint64(987654321)
    core_ratio = 0.6

    # Expected quotas from core/flex splits
    c_n = int(T_tiles * core_ratio)
    f_n = T_tiles - c_n
    rng16_core = _xs64star_rng_uint16(int(seed_core))
    rng16_flex = _xs64star_rng_uint16(int(seed_flex))
    tiles_c = sample_alias(prob, alias, rng16_core, c_n)
    tiles_f = sample_alias(prob, alias, rng16_flex, f_n)
    expected = {}
    for t in tiles_c.tolist() + tiles_f.tolist():
        expected[t] = expected.get(t, 0) + 1

    # Deterministic jitter rng
    rng_jit = _xs64star_rng_uint16(42)
    events = gen_block_events(
        pre_id=5,
        step=10,
        pre_tile=3,
        alias_for_tile=(prob, alias),
        tile_size=tile_size,
        T_tiles=T_tiles,
        M=M,
        seed_core=seed_core,
        seed_flex=seed_flex,
        core_ratio=core_ratio,
        delay_lut=np.zeros((ntiles, ntiles), dtype=np.uint8),
        rng=rng_jit,
    )

    # Check aggregated quotas and per-event shape
    total_k = 0
    seen_tiles = set()
    for ev in events:
        seen_tiles.add(ev.post_tile)
        assert ev.indices.size == M
        assert np.all(np.diff(ev.indices) >= 0)  # sorted
        q = expected.get(ev.post_tile, 0)
        assert q >= 1
        assert np.all(ev.k == q)
        total_k += int(q)
    assert len(seen_tiles) == len(events)
    assert total_k == T_tiles


def test_determinism_same_seed_step():
    ntiles = 6
    prob, alias = _uniform_alias(ntiles)
    tile_size = 32
    M = 8
    T_tiles = 20
    seed_core = np.uint64(555)
    seed_flex = np.uint64(777)
    core_ratio = 0.5

    args = dict(
        pre_id=9,
        step=42,
        pre_tile=4,
        alias_for_tile=(prob, alias),
        tile_size=tile_size,
        T_tiles=T_tiles,
        M=M,
        seed_core=seed_core,
        seed_flex=seed_flex,
        core_ratio=core_ratio,
        delay_lut=np.ones((ntiles, ntiles), dtype=np.uint8),
    )

    rng1 = _xs64star_rng_uint16(99)
    rng2 = _xs64star_rng_uint16(99)
    ev1 = gen_block_events(rng=rng1, **args)
    ev2 = gen_block_events(rng=rng2, **args)

    # Same seeds/rng -> identical
    assert len(ev1) == len(ev2)
    for a, b in zip(ev1, ev2):
        assert a.post_tile == b.post_tile
        assert a.delay == b.delay
        assert np.array_equal(a.indices, b.indices)
        assert np.array_equal(a.k, b.k)

    # Different step likely changes indices/delays deterministically
    rng3 = _xs64star_rng_uint16(99)
    ev3 = gen_block_events(rng=rng3, **{**args, "step": 43})
    different = (
        len(ev1) != len(ev3)
        or any(a.post_tile != b.post_tile for a, b in zip(ev1, ev3))
        or any(not np.array_equal(a.indices, b.indices) for a, b in zip(ev1, ev3))
        or any(a.delay != b.delay for a, b in zip(ev1, ev3))
    )
    assert different


def test_delay_from_lut_with_jitter_bounds():
    ntiles = 4
    prob, alias = _uniform_alias(ntiles)
    tile_size = 16
    M = 8
    T_tiles = 10
    seed_core = np.uint64(999)
    seed_flex = np.uint64(1001)
    core_ratio = 0.7

    # base delay = pre_tile + post_tile
    base = np.fromfunction(lambda i, j: i + j, (ntiles, ntiles), dtype=int).astype(np.uint8)

    # Jitter cycles 0..JITTER_MAX
    def rng_cycle(size: int) -> np.ndarray:
        seq = np.arange(size, dtype=np.uint16) % (JITTER_MAX + 1)
        return seq

    events = gen_block_events(
        pre_id=2,
        step=3,
        pre_tile=1,
        alias_for_tile=(prob, alias),
        tile_size=tile_size,
        T_tiles=T_tiles,
        M=M,
        seed_core=seed_core,
        seed_flex=seed_flex,
        core_ratio=core_ratio,
        delay_lut=base,
        rng=rng_cycle,
    )

    for ev in events:
        b = int(base[1, ev.post_tile])
        assert b <= ev.delay <= b + JITTER_MAX


def test_core_flex_split_respected():
    ntiles = 12
    prob, alias = _uniform_alias(ntiles)
    tile_size = 32
    M = 8
    T_tiles = 100
    core_ratio = 0.75
    seed_core = np.uint64(1)
    seed_flex = np.uint64(2)

    c_n = int(T_tiles * core_ratio)
    f_n = T_tiles - c_n
    rng16_core = _xs64star_rng_uint16(int(seed_core))
    rng16_flex = _xs64star_rng_uint16(int(seed_flex))
    tiles_c = sample_alias(prob, alias, rng16_core, c_n)
    tiles_f = sample_alias(prob, alias, rng16_flex, f_n)
    expected = {}
    for t in tiles_c.tolist() + tiles_f.tolist():
        expected[t] = expected.get(t, 0) + 1

    rng_jit = _xs64star_rng_uint16(7)
    events = gen_block_events(
        pre_id=0,
        step=0,
        pre_tile=0,
        alias_for_tile=(prob, alias),
        tile_size=tile_size,
        T_tiles=T_tiles,
        M=M,
        seed_core=seed_core,
        seed_flex=seed_flex,
        core_ratio=core_ratio,
        delay_lut=np.zeros((ntiles, ntiles), dtype=np.uint8),
        rng=rng_jit,
    )

    # Sum quotas and compare to expected dict; total sum equals T_tiles
    got = {}
    total = 0
    for ev in events:
        q = int(ev.k[0]) if ev.k.size else 0
        got[ev.post_tile] = q
        total += q
    assert total == T_tiles
    # Compare per-tile quotas (allow tiles absent in output if expected zero)
    for t, q in expected.items():
        if q == 0:
            assert t not in got or got[t] == 0
        else:
            assert got.get(t, 0) == q
