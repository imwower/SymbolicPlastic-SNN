import numpy as np

from symbolicplastic_snn.plasticity.stable_store import StableStore, StableEdge
from symbolicplastic_snn.conn.alias import build_alias
from symbolicplastic_snn.conn.generator import gen_block_events


def _uniform_alias_single(ntiles: int, focus: int):
    w = np.zeros(ntiles, dtype=np.uint16)
    w[focus] = np.uint16(65535)
    return build_alias(w)


def _rng_zero(size: int) -> np.ndarray:
    return np.zeros(size, dtype=np.uint16)


def test_add_get_remove():
    st = StableStore(per_pre_cap=2)
    # Add two edges ok
    assert st.add(StableEdge(pre_id=1, post_id=8, sign=np.int8(1), delay=np.uint8(0)))
    assert st.add(StableEdge(pre_id=1, post_id=9, sign=np.int8(1), delay=np.uint8(1)))
    # Exceed cap
    assert not st.add(StableEdge(pre_id=1, post_id=10, sign=np.int8(1), delay=np.uint8(0)))
    # Get & exists
    edges = st.get_pre(1)
    assert len(edges) == 2
    assert st.exists(1, 8, 0)
    assert st.exists(1, 9, 1)
    # Remove
    assert st.remove(1, 8, 0)
    assert not st.exists(1, 8, 0)


def test_injection_merging():
    ntiles = 2
    tile_size = 8
    pre_id = 0
    pre_tile = pre_id // tile_size
    focus_tile = 1  # inject towards tile 1
    prob, alias = _uniform_alias_single(ntiles, focus_tile)
    # Determine variable indices for this tile
    step = 3
    from symbolicplastic_snn.conn.permute import permute_first_m
    key = (pre_id * 0xDEAD) ^ ((focus_tile + 0x9E37) * 0xC2B2AE3D) ^ (step * 0x165667B1)
    M = 4
    var_indices = permute_first_m(tile_size, M, np.uint64(key))
    target_idx = int(var_indices[0])  # ensure overlap

    # Stable edge at delay 0 to the same (tile, idx)
    st = StableStore(per_pre_cap=8)
    post_id = focus_tile * tile_size + target_idx
    st.add(StableEdge(pre_id=pre_id, post_id=post_id, sign=np.int8(1), delay=np.uint8(0)))

    events = gen_block_events(
        pre_id=pre_id,
        step=step,
        pre_tile=pre_tile,
        alias_for_tile=(prob, alias),
        tile_size=tile_size,
        T_tiles=5,
        M=M,
        seed_core=np.uint64(1),
        seed_flex=np.uint64(2),
        core_ratio=1.0,
        delay_lut=np.zeros((ntiles, ntiles), dtype=np.uint8),
        rng=_rng_zero,
        stable_store=st,
    )

    # Single event towards focus_tile with delay 0
    evs = [e for e in events if e.post_tile == focus_tile and e.delay == 0]
    assert len(evs) == 1
    ev = evs[0]
    # Overlapped index should have k == quota + 1
    q = 5  # all samples go to the same tile
    # Find position of target_idx
    pos = int(np.where(ev.indices == target_idx)[0][0])
    assert int(ev.k[pos]) == (q + 1)


def test_budget_priority():
    ntiles = 2
    tile_size = 8
    pre_id = 0
    pre_tile = 0
    prob, alias = _uniform_alias_single(ntiles, 1)  # variable targets tile 1
    st = StableStore(per_pre_cap=8)
    # Inject three stable edges to tile 1 with delay 0
    for idx in [0, 1, 2]:
        st.add(StableEdge(pre_id=pre_id, post_id=1 * tile_size + idx, sign=np.int8(1), delay=np.uint8(0)))

    from symbolicplastic_snn.realtime.budgeter import EventBudget
    budget = EventBudget(b_step=3)  # only enough for stable

    events = gen_block_events(
        pre_id=pre_id,
        step=0,
        pre_tile=pre_tile,
        alias_for_tile=(prob, alias),
        tile_size=tile_size,
        T_tiles=10,
        M=5,
        seed_core=np.uint64(7),
        seed_flex=np.uint64(8),
        core_ratio=1.0,
        delay_lut=np.zeros((ntiles, ntiles), dtype=np.uint8),
        rng=_rng_zero,
        stable_store=st,
        budget=budget,
    )

    # Expect that only stable indices are present (budget used by them), no extra var-only indices
    evs = [e for e in events if e.post_tile == 1 and e.delay == 0]
    assert len(evs) == 1
    ev = evs[0]
    assert ev.indices.size == 3
