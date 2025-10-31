import numpy as np

from symbolicplastic_snn.conn.alias import build_alias, sample_alias, reweight_alias_smallstep


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


def _uniform_q016(n: int) -> np.ndarray:
    # Distribute 65535 uniformly across n entries.
    base = 65535 // n
    w = np.full(n, base, dtype=np.uint16)
    rem = 65535 - base * n
    if rem > 0:
        w[:rem] = (w[:rem].astype(np.uint32) + 1).astype(np.uint16)
    return w


def test_normalization_sum_65535():
    # Pseudo-random unnormalized weights using local xorshift64* helper
    rng16 = _xs64star_rng_uint16(123)
    raw = rng16(37)
    # Apply zero delta but enforce keep_sum; expect total 65535
    out = reweight_alias_smallstep(raw, np.zeros_like(raw, dtype=np.int32), keep_sum=True)
    assert out.dtype == np.uint16
    assert int(out.astype(np.uint32).sum()) == 65535


def test_uniform_sampling_frequency():
    n = 17
    w = _uniform_q016(n)
    prob, alias = build_alias(w)
    rng = _xs64star_rng_uint16(42)

    n_samples = 40000
    samples = sample_alias(prob, alias, rng, n_samples)

    # Frequency per bucket should be close to uniform within ±5%
    counts = np.bincount(samples, minlength=n)
    expected = n_samples / n
    tol = expected * 0.05
    for c in counts:
        assert abs(c - expected) <= tol


def test_reweight_keep_sum_and_clip():
    n = 21
    base = _uniform_q016(n)
    # Save copy for comparison
    before = base.copy()

    # Apply a positive delta to index 3 and negative to 7
    delta = np.zeros(n, dtype=np.int32)
    delta[3] = 2000
    delta[7] = -2000

    after = reweight_alias_smallstep(before, delta, keep_sum=True)
    assert int(after.astype(np.uint32).sum()) == 65535
    assert after[3] > before[3]
    assert after[7] < before[7]

    # Clipping: apply large negative/positive deltas
    delta2 = np.zeros(n, dtype=np.int32)
    delta2[0] = -100000  # should clip to 0 before normalization
    delta2[1] = 100000   # clip to 65535 before normalization
    clipped = reweight_alias_smallstep(before, delta2, keep_sum=False)
    assert clipped[0] == 0
    assert clipped[1] == 65535
