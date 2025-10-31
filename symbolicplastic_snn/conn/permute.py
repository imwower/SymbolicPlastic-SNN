from __future__ import annotations

import numpy as np


_MASK64 = (1 << 64) - 1


def _xs64star_next(state: int) -> tuple[int, int]:
    """xorshift64* generator step.

    Returns (new_state, rand64) with 64-bit wrapping arithmetic.
    xorshift64*: x ^= x >> 12; x ^= x << 25; x ^= x >> 27; return x * 2685821657736338717.
    """
    x = state & _MASK64
    x ^= (x >> 12) & _MASK64
    x ^= ((x << 25) & _MASK64)
    x ^= (x >> 27) & _MASK64
    new_state = x & _MASK64
    z = (new_state * 2685821657736338717) & _MASK64
    return new_state, z


def _u64_to_bounded(u: int, bound: int) -> int:
    """Map 64-bit uniform u in [0, 2^64) to [0, bound) without bias.

    Uses high-multiply technique: floor((u * bound) / 2^64).
    """
    return (u * bound) >> 64


def permute_first_m(n: int, m: int, key: np.uint64 | int) -> np.ndarray:
    """Deterministic permutation of first m slots via SplitMix64 + Fisher-Yates.

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

    # Initialize array [0, 1, ..., n-1]
    arr = np.arange(n, dtype=np.int32)
    # Initialize xorshift64* state from key
    if isinstance(key, np.generic):  # np.uint64 or similar
        state = int(key.item()) & _MASK64
    else:
        state = int(key) & _MASK64

    for i in range(m):
        state, u = _xs64star_next(state)
        bound = n - i
        j = i + _u64_to_bounded(u, bound)
        # swap arr[i], arr[j]
        ai = arr[i]
        aj = arr[j]
        arr[i] = aj
        arr[j] = ai

    return arr[:m]


__all__ = [
    "permute_first_m",
]
