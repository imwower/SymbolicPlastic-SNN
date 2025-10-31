from __future__ import annotations

import numpy as np


_MASK64 = (1 << 64) - 1


def _splitmix64_next(state: int) -> tuple[int, int]:
    """SplitMix64 next-state and output.

    Returns (new_state, output) using 64-bit wrapping arithmetic.
    """
    s = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = s
    z ^= (z >> 30) & _MASK64
    z = (z * 0xBF58476D1CE4E5B9) & _MASK64
    z ^= (z >> 27) & _MASK64
    z = (z * 0x94D049BB133111EB) & _MASK64
    z ^= (z >> 31) & _MASK64
    return s, z


def _u64_to_bounded(u: int, bound: int) -> int:
    """Map 64-bit uniform u in [0, 2^64) to [0, bound) without bias.

    Uses high-multiply technique: floor((u * bound) / 2^64).
    """
    return (u * bound) >> 64


def permute_first_m(n: int, m: int, key: np.uint64 | int) -> np.ndarray:
    """Deterministic permutation of first m slots via SplitMix64 + partial Fisher–Yates.

    Parameters
    - n: population size (>= 0)
    - m: number of positions to shuffle (0 <= m <= n)
    - key: 64-bit key for PRNG seeding (deterministic)

    Returns
    - int32 ndarray of length m with a permutation of unique indices in [0, n).
    """
    if n < 0 or m < 0 or m > n:
        raise ValueError("Require 0 <= m <= n and n >= 0")
    if n == 0 or m == 0:
        return np.zeros(0, dtype=np.int32)

    # Initialize SplitMix64 state from key
    if isinstance(key, np.generic):  # np.uint64 or similar
        state = int(key.item()) & _MASK64
    else:
        state = int(key) & _MASK64

    # Use O(m) mapping for partial Fisher–Yates without materializing full array
    swap: dict[int, int] = {}
    out = np.empty(m, dtype=np.int32)
    for i in range(m):
        state, u = _splitmix64_next(state)
        bound = n - i
        j = i + _u64_to_bounded(u, bound)
        # Resolve current values at i and j
        ai = swap.get(i, i)
        aj = swap.get(j, j)
        # Perform virtual swap
        swap[i] = aj
        swap[j] = ai
        out[i] = aj

    return out


__all__ = [
    "permute_first_m",
]
