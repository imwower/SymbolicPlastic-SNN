import numpy as np

from symbolicplastic_snn.schedule.timewheel import BlockEvent, TimeWheel


def be(post_tile, idx, k, delay):
    return BlockEvent(
        post_tile=post_tile,
        indices=np.asarray(idx, dtype=np.int32),
        k=np.asarray(k, dtype=np.int16),
        delay=delay,
    )


def test_push_pop_roundtrip():
    tw = TimeWheel(slots=4, bytes_cap=1 << 30)

    ev = be(3, [1, 5, 9], [2, 1, 3], delay=1)
    tw.push(ev)

    # Not yet due in current slot
    out0 = tw.pop()
    assert out0 == []

    # After one tick, event should appear
    tw.tick()
    out = tw.pop()
    assert len(out) == 1
    got = out[0]

    assert got.post_tile == 3
    assert got.delay == 1
    assert not got.capped
    assert got.total_k == 0
    assert np.array_equal(got.indices, np.array([1, 5, 9], dtype=np.int32))
    assert np.array_equal(got.k, np.array([2, 1, 3], dtype=np.int16))


def test_merge_same_tile_delay():
    tw = TimeWheel(slots=8, bytes_cap=1 << 30)

    ev1 = be(2, [1, 3, 5], [1, 2, 1], delay=2)
    ev2 = be(2, [3, 4], [2, 1], delay=2)
    ev3 = be(2, [1, 6], [3, 2], delay=2)

    tw.push_batch([ev1, ev2, ev3])

    # Not due yet
    assert tw.pop() == []
    tw.tick(); tw.tick()

    out = tw.pop()
    assert len(out) == 1
    got = out[0]
    assert got.post_tile == 2 and got.delay == 2 and not got.capped

    # indices must be strictly ascending and unique
    assert np.array_equal(got.indices, np.array([1, 3, 4, 5, 6], dtype=np.int32))
    # k aligned to indices: 1:(1+3)=4, 3:(2+2)=4, 4:1, 5:1, 6:2
    assert np.array_equal(got.k, np.array([4, 4, 1, 1, 2], dtype=np.int16))


def test_memory_cap_trigger():
    # Extremely low cap to force coarse aggregation
    tw = TimeWheel(slots=2, bytes_cap=8)
    # indices bytes = 5*4=20, k bytes = 5*2=10 -> 30 > cap
    ev = be(1, [0, 1, 2, 3, 4], [1, 2, 3, 4, 5], delay=0)
    tw.push(ev)

    out = tw.pop()
    assert len(out) == 1
    got = out[0]
    assert got.post_tile == 1 and got.delay == 0
    assert got.capped, "Should be coarse aggregated under cap"
    # Arrays should be empty when capped
    assert got.indices.size == 0 and got.k.size == 0
    # Total strength must equal sum of k
    assert int(got.total_k) == 1 + 2 + 3 + 4 + 5


def test_determinism():
    # Build a deterministic sequence of events with overlaps and different delays
    seq = []
    for t in range(5):
        # Three groups targeting different slots
        seq.append(be(0, [t, t + 1, t + 2], [1, 1, 1], delay=0))
        seq.append(be(0, [t + 1, t + 3], [2, 2], delay=1))
        seq.append(be(1, [2 * t, 2 * t + 1], [1, 3], delay=2))

    def run():
        tw = TimeWheel(slots=4, bytes_cap=1 << 30)
        outputs = []
        for i, ev in enumerate(seq):
            tw.push(ev)
            # Pop then tick per step for variety
            outs = tw.pop()
            # Canonicalize outputs: sorted by (tile, delay)
            outs = sorted(
                [
                    (
                        o.post_tile,
                        o.delay,
                        tuple(o.indices.tolist()),
                        tuple(o.k.tolist()),
                        bool(o.capped),
                        int(o.total_k),
                    )
                    for o in outs
                ]
            )
            outputs.append(tuple(outs))
            tw.tick()

        # Drain remaining slots
        for _ in range(4):
            outs = tw.pop()
            outs = sorted(
                [
                    (
                        o.post_tile,
                        o.delay,
                        tuple(o.indices.tolist()),
                        tuple(o.k.tolist()),
                        bool(o.capped),
                        int(o.total_k),
                    )
                    for o in outs
                ]
            )
            outputs.append(tuple(outs))
            tw.tick()
        return outputs

    out1 = run()
    out2 = run()
    assert out1 == out2, "Outputs must be deterministic for identical inputs"

