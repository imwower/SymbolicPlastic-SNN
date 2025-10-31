import numpy as np

from symbolicplastic_snn.schedule.timewheel import BlockEvent, TimeWheel


def be(post_tile, idx, k, delay):
    return BlockEvent(
        post_tile=int(post_tile),
        indices=np.asarray(idx, dtype=np.int32),
        k=np.asarray(k, dtype=np.int16),
        delay=int(delay),
    )


def test_roundtrip_single_event():
    tw = TimeWheel(slots=4)
    ev = be(3, [1, 5, 9], [2, 1, 4], 0)
    tw.push(ev)
    out = tw.pop()

    assert len(out) == 1
    o = out[0]
    assert o.post_tile == 3 and o.delay == 0
    assert not o.capped
    assert o.total_k == 0
    assert o.indices.dtype == np.int32 and o.k.dtype == np.int16
    assert o.indices.tolist() == [1, 5, 9]
    assert o.k.tolist() == [2, 1, 4]


def test_merge_same_tile_delay():
    tw = TimeWheel(slots=8)
    # Two events for same tile+delay with overlapping indices
    ev1 = be(1, [2, 4, 6], [1, 2, 3], 2)
    ev2 = be(1, [1, 4, 7], [4, 1, 1], 2)

    tw.push(ev1)
    tw.push(ev2)

    # Advance time until slot for delay=2
    tw.tick(); tw.tick()
    out = tw.pop()

    assert len(out) == 1
    o = out[0]
    assert o.post_tile == 1 and o.delay == 2
    # Merged and sorted indices; k aligned and summed on index 4
    assert o.indices.tolist() == [1, 2, 4, 6, 7]
    assert o.k.tolist() == [4, 1, 3, 3, 1]


def test_bytes_cap_degrade():
    # Very small cap to force degradation (arrays would be > cap)
    tw = TimeWheel(slots=2, bytes_cap=8)
    ev = be(0, [0, 1, 2, 3], [1, 1, 1, 1], 1)
    tw.push(ev)
    # Push another for same group to also accumulate total_k
    ev2 = be(0, [5, 6], [2, 3], 1)
    tw.push(ev2)

    tw.tick()
    out = tw.pop()

    assert len(out) == 1
    o = out[0]
    assert o.capped is True
    assert o.indices.size == 0 and o.k.size == 0
    # total_k equals sum of all k's pushed to this group
    assert int(o.total_k) == (4 * 1 + 2 + 3)


def test_determinism_sequence():
    slots = 4
    seq = [
        be(2, [3, 4], [1, 1], 0),
        be(1, [1, 2], [2, 3], 3),
        be(2, [4], [2], 0),
        be(2, [3], [5], 2),
        be(0, [10], [1], 1),
    ]

    def run_once():
        tw = TimeWheel(slots=slots)
        for ev in seq:
            tw.push(ev)
        outs = []
        for _ in range(slots):
            outs.append([(
                o.post_tile,
                o.delay,
                bool(o.capped),
                o.indices.copy(),
                o.k.copy(),
                int(o.total_k),
            ) for o in tw.pop()])
            tw.tick()
        return outs

    r1 = run_once()
    r2 = run_once()

    # Compare structure and contents deterministically
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert len(a) == len(b)
        for (pa, da, ca, ia, ka, ta), (pb, db, cb, ib, kb, tb) in zip(a, b):
            assert pa == pb and da == db and ca == cb and ta == tb
            assert np.array_equal(ia, ib)
            assert np.array_equal(ka, kb)

