import numpy as np

from symbolicplastic_snn.plasticity.update import reseed_small_fraction


def test_reseed_scope_core_unchanged():
    n = 256
    seeds_core = np.arange(n, dtype=np.uint64) * np.uint64(123456789)
    seeds_flex = np.arange(n, dtype=np.uint64) * np.uint64(987654321)
    seeds_core_before = seeds_core.copy()
    seeds_flex_before = seeds_flex.copy()

    # Only half are low contributors
    low = np.zeros(n, dtype=bool)
    low[::2] = True  # even indices eligible

    # With rate=1.0, all eligible should flip; ineligible unchanged
    reseed_small_fraction(seeds_core, seeds_flex, low, rate=1.0, epoch=42)

    # Core must not change
    assert np.array_equal(seeds_core, seeds_core_before)
    # Flex changed only where low is True
    changed = seeds_flex != seeds_flex_before
    assert np.all(changed[low])
    assert not np.any(changed[~low])


def test_reseed_rate_bounds():
    n = 4096
    seeds_core = np.arange(n, dtype=np.uint64) * np.uint64(0xD1342543)
    seeds_flex = np.arange(n, dtype=np.uint64) * np.uint64(0x94D049BB)
    low = np.ones(n, dtype=bool)  # all eligible

    seeds_flex_before = seeds_flex.copy()

    rate = 0.3
    reseed_small_fraction(seeds_core, seeds_flex, low, rate=rate, epoch=7)

    changed_frac = float(np.count_nonzero(seeds_flex != seeds_flex_before)) / n
    # Tolerance ±5%
    assert abs(changed_frac - rate) <= 0.05
