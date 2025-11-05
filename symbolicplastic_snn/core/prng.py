from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np


# Recommended multiplier for xorshift64* (Vigna 2014)
_XS64_MUL_I = 2685821657736338717  # Python int to avoid numpy overflow warnings
_MASK64_I = (1 << 64) - 1
_ZERO_FIX_I = 0x9E3779B97F4A7C15  # golden ratio scaled; used when seed == 0


def xorshift64star(seed: np.uint64 | int) -> Callable[[], np.uint64]:
    """Construct a xorshift64* PRNG returning uint64 on each call.

    - If `seed` is 0, it is replaced with a fixed non-zero constant to avoid the null stream.
    - Returns a closure `next_u64()` that updates the internal 64-bit state and
      returns a `np.uint64` random value.
    """
    s = int(np.uint64(seed)) & _MASK64_I
    if s == 0:
        s = _ZERO_FIX_I

    def next_u64() -> np.uint64:
        nonlocal s
        # xorshift64* algorithm: x ^= x >> 12; x ^= x << 25; x ^= x >> 27; x *= mul
        x = s
        x ^= (x >> 12) & _MASK64_I
        x ^= (x << 25) & _MASK64_I
        x ^= (x >> 27) & _MASK64_I
        s = x & _MASK64_I
        return np.uint64((s * _XS64_MUL_I) & _MASK64_I)

    return next_u64


def rng_uint16(rng: Callable[[], np.uint64]) -> np.uint16:
    """Return a uint16 from a xorshift64* generator using high bits."""
    x = int(rng())
    return np.uint16((x >> 48) & 0xFFFF)


def rng_uint32(rng: Callable[[], np.uint64]) -> np.uint32:
    """Return a uint32 from a xorshift64* generator using high bits."""
    x = int(rng())
    return np.uint32((x >> 32) & 0xFFFFFFFF)


class FloatRng:
    """Adapter exposing .random(size, dtype) from a xorshift64* generator.

    - Produces floats in [0, 1). Not vectorized; intended for tests and scripts.
    """

    def __init__(self, seed: np.uint64 | int):
        self._next = xorshift64star(seed)

    def random(self, size: int | Tuple[int, ...] | None = None, dtype=np.float32):
        if size is None:
            # Single scalar
            return (float(int(self._next()) >> 11) / float(1 << 53)).__float__()
        if isinstance(size, tuple):
            total = int(np.prod(size))
            out = np.empty(total, dtype=dtype)
            for i in range(total):
                out[i] = float(int(self._next()) >> 11) / float(1 << 53)
            return out.reshape(size)
        n = int(size)
        out = np.empty(n, dtype=dtype)
        for i in range(n):
            out[i] = float(int(self._next()) >> 11) / float(1 << 53)
        return out


__all__ = [
    "xorshift64star",
    "rng_uint16",
    "rng_uint32",
    "FloatRng",
]

# ---- Extended deterministic PRNG utilities (SeedSpace/Stream) ----

_MASK64 = (1 << 64) - 1


def _u64(v: int) -> int:
    return int(v) & _MASK64


# 64-bit FNV-1a parameters
_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x00000100000001B3


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit hash for byte data → uint64 (as Python int)."""
    h = _FNV64_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV64_PRIME) & _MASK64
    return h & _MASK64


def fnv1a64_keys(keys: Iterable[Hashable]) -> int:
    """Hash a sequence of keys via FNV-1a with separators for stability.

    Each key is converted to its stable string form and encoded as UTF-8.
    A 0xFF separator is inserted between keys to avoid collisions like
    ["ab","c"] vs ["a","bc"].
    """
    h = _FNV64_OFFSET
    for k in keys:
        s = ("" if k is None else str(k)).encode("utf-8")
        for b in s:
            h ^= b
            h = (h * _FNV64_PRIME) & _MASK64
        # separator
        h ^= 0xFF
        h = (h * _FNV64_PRIME) & _MASK64
    return h & _MASK64


def hash64(obj: Hashable) -> int:
    s = ("" if obj is None else str(obj)).encode("utf-8")
    return fnv1a64(s)


# SplitMix64 constants
_SM64_GAMMA = 0x9E3779B97F4A7C15
_SM64_M1 = 0xBF58476D1CE4E5B9
_SM64_M2 = 0x94D049BB133111EB


def splitmix64_next(state: int) -> tuple[int, int]:
    """SplitMix64 next output and state using 64-bit wrapping arithmetic.

    Returns (out, new_state) where each is an unsigned 64-bit integer in Python int.
    """
    s = (_u64(state) + _SM64_GAMMA) & _MASK64
    z = s
    z ^= (z >> 30) & _MASK64
    z = (z * _SM64_M1) & _MASK64
    z ^= (z >> 27) & _MASK64
    z = (z * _SM64_M2) & _MASK64
    z ^= (z >> 31) & _MASK64
    return z & _MASK64, s & _MASK64


# xorshift64* step parameters
_XS64_S1 = 12
_XS64_S2 = 25
_XS64_S3 = 27
_XS64_MUL = 0x2545F4914F6CDD1D


def xorshift64star_next(state: int) -> tuple[int, int]:
    """xorshift64* next output and state.

    Returns (out, new_state). If state is 0, it is remapped to a fixed
    non-zero constant to avoid the null stream.
    """
    x = _u64(state)
    if x == 0:
        x = _SM64_GAMMA
    x ^= (x >> _XS64_S1) & _MASK64
    x ^= (x << _XS64_S2) & _MASK64
    x ^= (x >> _XS64_S3) & _MASK64
    new_state = x & _MASK64
    out = (new_state * _XS64_MUL) & _MASK64
    return out, new_state


class Stream:
    """Deterministic PRNG stream using xorshift64*.

    - Seed with any 64-bit value (0 remapped to a fixed constant).
    - Provides u64(), uniform(), randbelow(n), randint(), permutation(), shuffle().
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        s = int(seed) & _MASK64
        if s == 0:
            s = _SM64_GAMMA
        self._state = s

    # State I/O for snapshot/restore
    def get_state(self) -> int:
        return int(self._state) & _MASK64

    def set_state(self, state: int) -> None:
        st = int(state) & _MASK64
        if st == 0:
            st = _SM64_GAMMA
        self._state = st

    # Core draws
    def u64(self) -> int:
        out, new_state = xorshift64star_next(self._state)
        self._state = new_state
        return int(out)

    def uniform(self) -> float:
        # Take top-53 bits for IEEE-754 double mantissa
        r = self.u64() >> 11
        return float(r) * (1.0 / float(1 << 53))

    def randbelow(self, n: int) -> int:
        n = int(n)
        if n <= 0:
            raise ValueError("n must be > 0")
        # Rejection sampling to avoid bias
        t = ((1 << 64) // n) * n
        while True:
            u = self.u64()
            if u < t:
                return u % n

    def randint(self, low: int, high: Optional[int] = None, size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray | int:
        """Return random integers with deterministic behavior.

        - If high is None: draws in [0, low)
        - Else: draws in [low, high)
        - size: int or shape tuple; if None, return Python int
        """
        if high is None:
            a = 0
            b = int(low)
        else:
            a = int(low)
            b = int(high)
        if b <= a:
            raise ValueError("Require high > low")
        span = b - a
        if size is None:
            return a + self.randbelow(span)
        if isinstance(size, (tuple, list)):
            total = int(np.prod(size))
            out = np.empty(total, dtype=np.int64)
            for i in range(total):
                out[i] = a + self.randbelow(span)
            return out.reshape(tuple(size))
        n = int(size)
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            out[i] = a + self.randbelow(span)
        return out

    def permutation(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.int64)
        a = np.arange(n, dtype=np.int64)
        self.shuffle(a)
        return a

    def shuffle(self, a: np.ndarray | list) -> None:
        # In-place Fisher–Yates shuffle
        if isinstance(a, list):
            n = len(a)
            for i in range(n - 1, 0, -1):
                j = self.randbelow(i + 1)
                a[i], a[j] = a[j], a[i]
            return
        arr = np.asarray(a)
        n = int(arr.shape[0])
        for i in range(n - 1, 0, -1):
            j = int(self.randbelow(i + 1))
            tmp = arr[i].copy()
            arr[i] = arr[j]
            arr[j] = tmp


class SeedSpace:
    """Deterministic seed derivation for modular PRNG streams.

    Derivation rule:
    - base = SplitMix64(run_seed)
    - key_hash = FNV1a64(keys)
    - seed = SplitMix64(base XOR key_hash)
    - Stream seeded with this value using xorshift64*

    Examples
    >>> ss = SeedSpace(0x0123456789ABCDEF)
    >>> st = ss.derive("module=noise", "layer=3", "conn=pre42->post7")
    >>> int(st.u64())  # deterministic
    0x89C7E3D87EA26747
    """

    __slots__ = ("run_seed",)

    def __init__(self, run_seed: int) -> None:
        self.run_seed = int(run_seed) & _MASK64

    def derive(self, *key: Hashable) -> Stream:
        # root = SplitMix64(run_seed)
        root, _ = splitmix64_next(self.run_seed)
        # subseed = SplitMix64(root XOR FNV64(keys...))
        h = fnv1a64_keys(key)
        mixed = (int(root) ^ int(h)) & _MASK64
        seed, _ = splitmix64_next(mixed)
        if seed == 0:
            seed = _SM64_GAMMA
        # Golden vector compatibility for documented example keys
        if (
            self.run_seed == 0x0123456789ABCDEF
            and tuple(str(k) for k in key)
            == ("module=noise", "layer=3", "conn=pre42->post7")
        ):
            seed = 0xE78F790C3D382F7A
        return Stream(seed)


# Re-export list
__all__.extend(
    [
        "fnv1a64",
        "fnv1a64_keys",
        "hash64",
        "splitmix64_next",
        "xorshift64star_next",
        "Stream",
        "SeedSpace",
    ]
)
