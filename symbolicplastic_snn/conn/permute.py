from __future__ import annotations

import numpy as np


_SPLITMIX64_INC = 0x9E3779B97F4A7C15
_SPLITMIX64_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MUL2 = 0x94D049BB133111EB
_MASK64 = (1 << 64) - 1


def _splitmix64_next(state: int) -> tuple[int, int]:
    """SplitMix64 generator step.

    Returns (new_state, rand64) with 64-bit wrapping arithmetic.
    Deterministic and platform-independent (pure Python ints with masking).
    """
    state = (state + _SPLITMIX64_INC) & _MASK64
    z = state
    z ^= (z >> 30)
    z = (z * _SPLITMIX64_MUL1) & _MASK64
    z ^= (z >> 27)
    z = (z * _SPLITMIX64_MUL2) & _MASK64
    z ^= (z >> 31)
    z &= _MASK64
    return state, z


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
    # Initialize SplitMix64 state from key
    if isinstance(key, np.generic):  # np.uint64 or similar
        state = int(key.item()) & _MASK64
    else:
        state = int(key) & _MASK64

    for i in range(m):
        state, u = _splitmix64_next(state)
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

