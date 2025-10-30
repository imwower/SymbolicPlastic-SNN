import numpy as np

from symbolicplastic_snn.conn.alias import build_alias, sample_alias, reweight_alias


def _splitmix64_rng_uint16(seed: int):
    MASK = (1 << 64) - 1
    INC = 0x9E3779B97F4A7C15
    MUL1 = 0xBF58476D1CE4E5B9
    MUL2 = 0x94D049BB133111EB

    state = int(seed) & MASK

    def next_u64():
        nonlocal state
        state = (state + INC) & MASK
        z = state
        z ^= (z >> 30)
        z = (z * MUL1) & MASK
        z ^= (z >> 27)
        z = (z * MUL2) & MASK
        z ^= (z >> 31)
        return z & MASK

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


def test_alias_sum_q016():
    # Random unnormalized weights
    rng = np.random.default_rng(123)
    raw = rng.integers(low=0, high=65535, size=37, dtype=np.uint16)
    # Apply zero delta but enforce keep_sum; expect total 65535
    out = reweight_alias(raw, np.zeros_like(raw, dtype=np.int32), keep_sum=True)
    assert out.dtype == np.uint16
    assert int(out.astype(np.uint32).sum()) == 65535


def test_alias_sampling_frequency():
    n = 17
    w = _uniform_q016(n)
    prob, alias = build_alias(w)
    rng = _splitmix64_rng_uint16(42)

    n_samples = 40000
    samples = sample_alias(prob, alias, rng, n_samples)

    # Frequency per bucket should be close to uniform within ±5%
    counts = np.bincount(samples, minlength=n)
    expected = n_samples / n
    tol = expected * 0.05
    for c in counts:
        assert abs(c - expected) <= tol


def test_reweight_keep_sum():
    n = 21
    base = _uniform_q016(n)
    # Save copy for comparison
    before = base.copy()

    # Apply a positive delta to index 3 and negative to 7
    delta = np.zeros(n, dtype=np.int32)
    delta[3] = 2000
    delta[7] = -2000

    after = reweight_alias(before, delta, keep_sum=True)
    assert int(after.astype(np.uint32).sum()) == 65535
    assert after[3] > before[3]
    assert after[7] < before[7]

