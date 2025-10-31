import numpy as np

from symbolicplastic_snn.core.prng import xorshift64star, rng_uint16, rng_uint32


def take_stream_u64(seed: int, n: int):
    rng = xorshift64star(np.uint64(seed))
    return [int(rng()) for _ in range(n)]


def test_repeatability():
    n = 100
    s = 123456789
    a = take_stream_u64(s, n)
    b = take_stream_u64(s, n)
    c = take_stream_u64(s + 1, n)

    assert a == b  # same seed -> identical stream
    assert a != c  # different seed -> different stream (very likely)

    # Check helper projections are consistent
    rng1 = xorshift64star(np.uint64(s))
    rng2 = xorshift64star(np.uint64(s))
    u16_1 = [int(rng_uint16(rng1)) for _ in range(n)]
    u16_2 = [int(rng_uint16(rng2)) for _ in range(n)]
    assert u16_1 == u16_2

    rng3 = xorshift64star(np.uint64(s))
    rng4 = xorshift64star(np.uint64(s))
    u32_1 = [int(rng_uint32(rng3)) for _ in range(n)]
    u32_2 = [int(rng_uint32(rng4)) for _ in range(n)]
    assert u32_1 == u32_2


def test_nonzero_lock():
    # Seed 0 should be remapped to a fixed non-zero state
    a = take_stream_u64(0, 16)
    b = take_stream_u64(0, 16)
    assert a == b
    # Not all zeros; stream produces varying values
    assert any(x != 0 for x in a)

