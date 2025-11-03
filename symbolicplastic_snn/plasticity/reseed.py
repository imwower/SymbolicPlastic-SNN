from __future__ import annotations

import numpy as np

from .update import _xs64star_mix  # reuse deterministic mixer


def reseed_small_fraction(
    seeds_flex: np.ndarray,
    mask: np.ndarray,
    rate: float = 0.01,
    epoch: int = 0,
) -> int:
    """Reseed a small fraction of exploration seeds (flex only).

    - seeds_flex: uint64 array, mutated in-place where selected
    - mask: boolean array marking eligible indices
    - rate: fraction in [0,1]
    - epoch: integer used to derive deterministic hash threshold
    Returns number of reseeded entries.
    """
    sf = np.asarray(seeds_flex, dtype=np.uint64)
    m = np.asarray(mask, dtype=bool)
    if sf.shape[0] != m.shape[0]:
        raise ValueError("seeds_flex and mask must have same length")
    n = sf.shape[0]
    if n == 0:
        return 0
    r = float(rate)
    if r <= 0.0 or not np.any(m):
        return 0

    idx = np.arange(n, dtype=np.uint64)
    eph = np.uint64(_xs64star_mix(int(epoch)))
    mixed = (sf ^ (idx * np.uint64(0xD1342543DE82EF95)) ^ eph).astype(np.uint64)
    # Vectorized hash over mixed values
    mixed_h = np.fromiter((_xs64star_mix(int(v)) for v in mixed), count=n, dtype=np.uint64)
    threshold = np.uint64(int(max(0.0, min(1.0, r)) * (1 << 64)))
    sel = m & (mixed_h < threshold)
    if not np.any(sel):
        return 0
    # Flip with epoch hash
    sf[sel] = (sf[sel] ^ eph).astype(np.uint64)
    seeds_flex[...] = sf
    return int(np.count_nonzero(sel))


__all__ = ["reseed_small_fraction"]

