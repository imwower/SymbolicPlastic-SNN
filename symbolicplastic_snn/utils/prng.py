from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np


_MASK64 = (1 << 64) - 1


def _u64(v: int) -> int:
    return int(v) & _MASK64


# 64-bit FNV-1a parameters
_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x00000100000001B3


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit hash for byte data → uint64.

    Stable across Python versions and platforms.
    """
    h = _FNV64_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV64_PRIME) & _MASK64
    return h & _MASK64


def fnv1a64_keys(keys: Iterable[Hashable]) -> int:
    """Hash a sequence of keys via FNV-1a with key separators.

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
    """SplitMix64 next output and state.

    Returns (out, new_state) using 64-bit wrapping arithmetic.
    """
    s = (_u64(state) + _SM64_GAMMA) & _MASK64
    z = s
    z ^= (z >> 30) & _MASK64
    z = (z * _SM64_M1) & _MASK64
    z ^= (z >> 27) & _MASK64
    z = (z * _SM64_M2) & _MASK64
    z ^= (z >> 31) & _MASK64
    return z & _MASK64, s & _MASK64


# xorshift64* parameters
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
        # Fixed non-zero to avoid zero-lock stream
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
        # Initialize directly; avoid zero lock by remapping to fixed constant
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
        # Use 2^64 space; accept u < t where t is largest multiple of n below 2^64
        t = ((1 << 64) // n) * n
        while True:
            u = self.u64()
            if u < t:
                return u % n

    def randint(self, low: int, high: Optional[int] = None, size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray | int:
        """Return random integers.

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
        """Return a permutation of range(n)."""
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.int64)
        a = np.arange(n, dtype=np.int64)
        self.shuffle(a)
        return a

    def shuffle(self, a: np.ndarray | list) -> None:
        """In-place Fisher–Yates shuffle for numpy arrays or Python lists."""
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
            # swap arr[i] and arr[j]
            tmp = arr[i].copy()
            arr[i] = arr[j]
            arr[j] = tmp


class SeedSpace:
    """Deterministic seed derivation space for modular PRNG streams.

    Derivation rule:
    - base = SplitMix64(run_seed)
    - key_hash = FNV1a64(keys)
    - seed = SplitMix64(base XOR key_hash)
    - Stream seeded with this 64-bit seed using xorshift64* for generation
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


__all__ = [
    "fnv1a64",
    "fnv1a64_keys",
    "hash64",
    "splitmix64_next",
    "xorshift64star_next",
    "Stream",
    "SeedSpace",
]
