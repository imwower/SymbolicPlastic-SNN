from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np


# Recommended multiplier for xorshift64* (Vigna 2014)
_XS64_MUL = np.uint64(2685821657736338717)
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_ZERO_FIX = np.uint64(0x9E3779B97F4A7C15)  # golden ratio scaled; used when seed == 0


def xorshift64star(seed: np.uint64 | int) -> Callable[[], np.uint64]:
    """Construct a xorshift64* PRNG returning uint64 on each call.

    - If `seed` is 0, it is replaced with a fixed non-zero constant to avoid the null stream.
    - Returns a closure `next_u64()` that updates the internal 64-bit state and
      returns a `np.uint64` random value.
    """
    s = np.uint64(seed)
    if s == np.uint64(0):
        s = _ZERO_FIX

    def next_u64() -> np.uint64:
        nonlocal s
        # xorshift64* algorithm: x ^= x >> 12; x ^= x << 25; x ^= x >> 27; x *= mul
        x = s
        x ^= (x >> np.uint64(12)) & _MASK64
        x ^= (x << np.uint64(25)) & _MASK64
        x ^= (x >> np.uint64(27)) & _MASK64
        s = x & _MASK64
        return (s * _XS64_MUL) & _MASK64

    return next_u64


def rng_uint16(rng: Callable[[], np.uint64]) -> np.uint16:
    """Return a uint16 from a xorshift64* generator using high bits."""
    x = rng()
    return np.uint16((int(x) >> 48) & 0xFFFF)


def rng_uint32(rng: Callable[[], np.uint64]) -> np.uint32:
    """Return a uint32 from a xorshift64* generator using high bits."""
    x = rng()
    return np.uint32((int(x) >> 32) & 0xFFFFFFFF)


class FloatRng:
    """Adapter exposing .random(size, dtype) from a xorshift64* generator.

    - Produces floats in [0, 1). Not vectorized; intended for tests and scripts.
    """

    def __init__(self, seed: np.uint64 | int):
        self._next = xorshift64star(seed)

    def random(self, size: int | Tuple[int, ...] | None = None, dtype=np.float32):
        if size is None:
            # Single scalar
            return (float(int(self._next() >> np.uint64(11)) / float(1 << 53))).__float__()
        if isinstance(size, tuple):
            total = int(np.prod(size))
            out = np.empty(total, dtype=dtype)
            for i in range(total):
                out[i] = float(int(self._next() >> np.uint64(11)) / float(1 << 53))
            return out.reshape(size)
        n = int(size)
        out = np.empty(n, dtype=dtype)
        for i in range(n):
            out[i] = float(int(self._next() >> np.uint64(11)) / float(1 << 53))
        return out


__all__ = [
    "xorshift64star",
    "rng_uint16",
    "rng_uint32",
    "FloatRng",
]

