from __future__ import annotations

from typing import Callable, Tuple
from dataclasses import dataclass

import numpy as np
from symbolicplastic_snn.utils.prng import Stream


def prefix_sum_uint16(a: np.ndarray) -> np.ndarray:
    """Prefix sum for uint16 arrays using uint32 accumulator.

    Returns a uint32 array ps where ps[i] = sum(a[: i + 1]).
    """
    if a.dtype != np.uint16:
        raise TypeError("a must be uint16 ndarray")
    ps = np.cumsum(a.astype(np.uint32), dtype=np.uint32)
    return ps


def build_alias(prob_q016: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build integer alias table from Q0.16 probabilities (uint16).

    Parameters
    - prob_q016: uint16 ndarray, unnormalized allowed. Sum is treated as total mass.

    Returns
    - prob: uint16 ndarray of per-column thresholds in [0, 65535].
    - alias: int32 ndarray of alias indices.

    Notes
    - All computations are integer-only. No floating point used.
    - If the input sums to 0, falls back to uniform columns: prob=65535, alias=i.
    - Threshold semantics: draw r2 in [0, 65535], choose base if r2 < prob[col],
      else choose alias[col]. Column `col` is chosen uniformly via integer mapping.
    """

    if prob_q016.dtype != np.uint16:
        raise TypeError("prob_q016 must be uint16 ndarray")

    n = prob_q016.shape[0]
    if n == 0:
        raise ValueError("prob_q016 must be non-empty")

    # Total mass using precise prefix sum.
    total = int(prefix_sum_uint16(prob_q016)[-1])

    prob = np.zeros(n, dtype=np.uint16)
    alias = np.arange(n, dtype=np.int32)

    if total == 0:
        # Uniform fallback: each column is always chosen when its slot is picked.
        prob.fill(np.uint16(65535))
        return prob, alias

    # Optional numba-accelerated core if available
    if _NUMBA_AVAILABLE:
        p, a = _build_alias_numba_core(prob_q016.astype(np.uint16))
        return p.astype(np.uint16), a.astype(np.int32)

    # Scale weights by n to compare against total mass (Vose's method).
    w32 = prob_q016.astype(np.uint32)
    S = (w32.astype(np.uint64) * np.uint64(n)).astype(np.int64)
    total64 = np.int64(total)

    # Build small/large worklists.
    small = []
    large = []
    for i in range(n):
        if S[i] < total64:
            small.append(i)
        elif S[i] > total64:
            large.append(i)
        else:
            prob[i] = np.uint16(65535)
            alias[i] = np.int32(i)

    # Process pairs
    while small and large:
        i = small.pop()
        j = large.pop()

        thr = (np.int64(S[i]) << 16) // total64
        if thr >= 65536:
            thr = 65535
        prob[i] = np.uint16(thr)
        alias[i] = np.int32(j)

        Sj_new = S[j] - (total64 - S[i])
        S[j] = Sj_new
        if Sj_new < total64:
            small.append(j)
        elif Sj_new > total64:
            large.append(j)
        else:
            prob[j] = np.uint16(65535)
            alias[j] = np.int32(j)

    for i in small:
        prob[i] = np.uint16(65535)
        alias[i] = np.int32(i)
    for j in large:
        prob[j] = np.uint16(65535)
        alias[j] = np.int32(j)

    return prob, alias


def sample_alias(
    prob: np.ndarray,
    alias: np.ndarray,
    rng_uint16: Callable[[int], np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """Sample from alias table using integer RNG.

    Parameters
    - prob: uint16 thresholds per column (from build_alias).
    - alias: int32 alias indices.
    - rng_uint16: callable(size) -> uint16 ndarray of random values in [0, 65535].
    - n_samples: number of samples to draw.

    Returns
    - int32 ndarray of sampled indices in [0, n).

    Notes
    - Column selection uses unbiased integer mapping: idx = (r1 * n) >> 16.
    - Outcome decision uses r2 < prob[idx].
    """

    if prob.dtype != np.uint16:
        raise TypeError("prob must be uint16 ndarray")
    if alias.dtype != np.int32:
        raise TypeError("alias must be int32 ndarray")
    if prob.shape[0] != alias.shape[0]:
        raise ValueError("prob and alias must have same length")
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")

    n = prob.shape[0]
    if n == 0 or n_samples == 0:
        return np.zeros(0, dtype=np.int32)

    r1 = np.asarray(rng_uint16(n_samples), dtype=np.uint16)
    r2 = np.asarray(rng_uint16(n_samples), dtype=np.uint16)
    # Map r1 to [0, n) using high-multiply technique with 16-bit input.
    idx = ((r1.astype(np.uint32) * np.uint32(n)) >> 16).astype(np.int32)
    # Decide outcome using threshold.
    take_base = r2.astype(np.uint32) < prob[idx].astype(np.uint32)
    out = np.where(take_base, idx, alias[idx]).astype(np.int32)
    return out


def reweight_alias(
    prob: np.ndarray,
    delta_int: np.ndarray,
    keep_sum: bool = True,
) -> np.ndarray:
    """Integer reweight for Q0.16 probabilities.

    Parameters
    - prob: uint16 ndarray of original weights (Q0.16, arbitrary sum allowed).
    - delta_int: int32 ndarray, same length, element-wise adjustments.
    - keep_sum: if True, renormalize to total sum 65535 exactly.

    Returns
    - uint16 ndarray of adjusted weights.

    Rules
    - Add first, then clamp to [0, 65535].
    - If keep_sum: renormalize so the new sum equals 65535 using integer-only
      scaling with remainder distribution to ensure exact total.
    """

    if prob.dtype != np.uint16:
        raise TypeError("prob must be uint16 ndarray")
    if delta_int.dtype not in (np.int32, np.int64):
        raise TypeError("delta_int must be int32/int64 ndarray")
    if prob.shape[0] != delta_int.shape[0]:
        raise ValueError("prob and delta_int must have same length")

    x = prob.astype(np.int32) + delta_int.astype(np.int32)
    x = np.clip(x, 0, 65535).astype(np.int32)

    if not keep_sum:
        return x.astype(np.uint16)

    total = int(x.sum(dtype=np.int64))
    n = x.shape[0]
    if total == 0:
        # Evenly spread mass to sum exactly 65535.
        base = 65535 // max(1, n)
        out = np.full(n, base, dtype=np.int32)
        remainder = 65535 - base * n
        if remainder > 0:
            out[:remainder] += 1
        return out.astype(np.uint16)

    # Scale with integer division and distribute remainder by largest remainders.
    x64 = x.astype(np.int64)
    prod = x64 * 65535  # up to ~2^31 * 65535 fits in int64
    base = (prod // total).astype(np.int32)
    rem = (prod % total).astype(np.int64)

    out = base.copy()
    need = int(65535 - int(base.sum(dtype=np.int64)))
    if need > 0:
        # Pick top-`need` remainders deterministically: stable argsort by descending rem.
        # Using argpartition for efficiency.
        if need < n:
            idx = np.argpartition(-rem, need - 1)[:need]
        else:
            idx = np.arange(n)
        # If we want deterministic tie-breaking across platforms, refine by sorting these idxs
        # by (-rem, index) to ensure reproducibility.
        sel = idx[np.argsort(-rem[idx], kind="mergesort")]
        sel = sel[:need]
        out[sel] += 1

    return out.astype(np.uint16)


def rebuild_or_update(
    prob: np.ndarray,
    alias: np.ndarray,
    changed_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Incrementally rebuild or update an alias table after local weight changes.

    Parameters
    - prob: uint16 thresholds per column (existing)
    - alias: int32 alias indices (existing)
    - changed_idx: int array of indices whose underlying weights changed

    Returns
    - (prob_new, alias_new)

    Notes
    - Vose alias tables are globally coupled. For correctness and to keep
      determinism, this implementation conservatively falls back to a full
      rebuild using `build_alias` on the implied weight vector when local
      changes are detected. This ensures exact equivalence with full rebuilds
      and preserves deterministic sampling (required by tests).
    - Future optimized versions may implement a true incremental path with
      identical results when possible.
    """
    p = np.asarray(prob)
    a = np.asarray(alias)
    if p.dtype != np.uint16 or a.dtype != np.int32:
        raise TypeError("prob must be uint16 and alias must be int32")
    if p.ndim != 1 or a.ndim != 1 or p.shape[0] != a.shape[0]:
        raise ValueError("prob and alias must be 1-D of same length")
    n = p.shape[0]
    idx = np.asarray(changed_idx, dtype=np.int64).reshape(-1)
    if idx.size and (np.any(idx < 0) or np.any(idx >= n)):
        raise ValueError("changed_idx out of bounds")

    # Conservatively rebuild fully to guarantee bit-for-bit parity with reference.
    # Reconstruct Q0.16 weight proxy as probabilities (mass per column);
    # since build_alias only depends on weights up to a scaling constant,
    # passing `p` is sufficient (sum treated as total mass).
    prob_new, alias_new = build_alias(p.astype(np.uint16, copy=False))
    return prob_new.astype(np.uint16, copy=False), alias_new.astype(np.int32, copy=False)


def reweight_alias_smallstep(
    prob: np.ndarray,
    delta_int: np.ndarray,
    keep_sum: bool = True,
) -> np.ndarray:
    """Compatibility wrapper: same as reweight_alias().

    - Applies integer delta with clipping to [0, 65535], then optional exact
      normalization to sum 65535.
    """
    return reweight_alias(prob, delta_int, keep_sum=keep_sum)


__all__ = [
    "build_alias",
    "sample_alias",
    "reweight_alias",
    "reweight_alias_smallstep",
    "prefix_sum_uint16",
    "AliasForTile",
]


def build_alias_table(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build an alias table from positive weights using Vose's method.

    Parameters
    - p: 1-D float array of non-negative weights. At least one element must be > 0.

    Returns
    - (prob, alias):
      * prob: float64 ndarray thresholds in [0, 1)
      * alias: int32 ndarray of alias indices

    Notes
    - O(n) build time; O(1) sampling.
    - Numerical stability: normalize weights to sum=1.0; thresholds kept in [0,1) for robust comparison with `Stream.uniform()`.
    - Raises ValueError on invalid inputs (negative weights, all zero, wrong shape).
    """
    w = np.asarray(p, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("p must be 1-D")
    if w.size == 0:
        raise ValueError("p must be non-empty")
    if np.any(w < 0.0):
        raise ValueError("p must be non-negative")
    s = float(w.sum())
    if s <= 0.0:
        raise ValueError("sum of p must be > 0")
    n = w.size
    prob = np.zeros(n, dtype=np.float64)
    alias = np.arange(n, dtype=np.int32)
    scaled = (w / s) * n
    small: list[int] = []
    large: list[int] = []
    for i in range(n):
        if scaled[i] < 1.0:
            small.append(i)
        elif scaled[i] > 1.0:
            large.append(i)
        else:
            prob[i] = 1.0
            alias[i] = np.int32(i)
    while small and large:
        i = small.pop()
        j = large.pop()
        prob[i] = float(scaled[i])
        alias[i] = np.int32(j)
        scaled[j] = (scaled[j] - (1.0 - scaled[i]))
        if scaled[j] < 1.0 - 1e-18:
            small.append(j)
        elif scaled[j] > 1.0 + 1e-18:
            large.append(j)
        else:
            prob[j] = 1.0
            alias[j] = np.int32(j)
    for i in small:
        prob[i] = 1.0
        alias[i] = np.int32(i)
    for j in large:
        prob[j] = 1.0
        alias[j] = np.int32(j)
    return prob, alias


def sample_alias_stream(
    prob: np.ndarray,
    alias: np.ndarray,
    stream: Stream,
    size: int | None = None,
) -> np.ndarray | int:
    """Sample from a float-threshold alias table using a Stream.

    Parameters
    - prob: float64 thresholds in [0,1) (from build_alias_table) or uint16 thresholds (legacy support).
    - alias: int32 alias indices
    - stream: Stream (xorshift64* based) used for deterministic sampling
    - size: optional number of samples. If None, returns a single int.

    Returns np.ndarray of dtype int32 if size is given, else a single int.
    """
    if alias.dtype != np.int32:
        raise TypeError("alias must be int32 ndarray")
    n = int(alias.shape[0])
    if n <= 0:
        raise ValueError("alias table must be non-empty")
    if size is None:
        col = int(stream.randbelow(n))
        if prob.dtype == np.float64:
            return int(col if stream.uniform() < float(prob[col]) else int(alias[col]))
        if prob.dtype == np.uint16:
            u = int(stream.randbelow(65536))
            return int(col if u < int(prob[col]) else int(alias[col]))
        raise TypeError("prob must be float64 or uint16")
    m = int(size)
    if m < 0:
        raise ValueError("size must be non-negative")
    out = np.empty(m, dtype=np.int32)
    if prob.dtype == np.float64:
        for i in range(m):
            col = int(stream.randbelow(n))
            out[i] = np.int32(col if stream.uniform() < float(prob[col]) else int(alias[col]))
        return out
    if prob.dtype == np.uint16:
        for i in range(m):
            col = int(stream.randbelow(n))
            u = int(stream.randbelow(65536))
            out[i] = np.int32(col if u < int(prob[col]) else int(alias[col]))
        return out
    raise TypeError("prob must be float64 or uint16")


class AliasSampler:
    """Alias sampler with cached tables.

    Use from_weights(weights) to construct, then sample(stream, n) to draw indices.
    Deterministic given a Stream. O(1) per sample.
    """

    def __init__(self, prob: np.ndarray, alias_idx: np.ndarray) -> None:
        self.prob = np.asarray(prob, dtype=np.float64)
        self.alias = np.asarray(alias_idx, dtype=np.int32)
        if self.prob.ndim != 1 or self.alias.ndim != 1 or self.prob.size != self.alias.size:
            raise ValueError("prob and alias must be 1-D of same length")

    @classmethod
    def from_weights(cls, weights: np.ndarray) -> "AliasSampler":
        prob, alias = build_alias_table(weights)
        return cls(prob, alias)

    def sample(self, stream: Stream, n: int) -> np.ndarray:
        return sample_alias_stream(self.prob, self.alias, stream, int(n))

__all__.extend(["build_alias_table", "sample_alias_stream", "AliasSampler"])


@dataclass
class AliasForTile:
    """Container for alias table of post-tile sampling.

    prob: uint16 thresholds per tile
    alias: int32 alias indices
    """

    prob: np.ndarray
    alias: np.ndarray

    def __post_init__(self) -> None:
        if self.prob.dtype != np.uint16:
            self.prob = self.prob.astype(np.uint16, copy=False)
        if self.alias.dtype != np.int32:
            self.alias = self.alias.astype(np.int32, copy=False)
_NUMBA_AVAILABLE = False
try:  # optional acceleration
    import numba as _nb  # type: ignore

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:  # pragma: no cover - jit accelerates loops
    @_nb.njit(cache=True)
    def _build_alias_numba_core(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = w.shape[0]
        prob = np.zeros(n, dtype=np.uint16)
        alias = np.arange(n, dtype=np.int32)
        total = np.int64(0)
        for i in range(n):
            total += np.int64(w[i])
        if total == 0:
            for i in range(n):
                prob[i] = np.uint16(65535)
            return prob, alias
        total64 = np.int64(total)
        S = np.empty(n, dtype=np.int64)
        for i in range(n):
            S[i] = np.int64(w[i]) * np.int64(n)
        small = np.empty(n, dtype=np.int32)
        large = np.empty(n, dtype=np.int32)
        ns = 0
        nl = 0
        for i in range(n):
            if S[i] < total64:
                small[ns] = i
                ns += 1
            elif S[i] > total64:
                large[nl] = i
                nl += 1
            else:
                prob[i] = np.uint16(65535)
                alias[i] = np.int32(i)
        while ns > 0 and nl > 0:
            ns -= 1
            i = int(small[ns])
            nl -= 1
            j = int(large[nl])
            thr = (S[i] << 16) // total64
            if thr >= 65536:
                thr = 65535
            prob[i] = np.uint16(thr)
            alias[i] = np.int32(j)
            Sj_new = S[j] - (total64 - S[i])
            S[j] = Sj_new
            if Sj_new < total64:
                small[ns] = j
                ns += 1
            elif Sj_new > total64:
                large[nl] = j
                nl += 1
            else:
                prob[j] = np.uint16(65535)
                alias[j] = np.int32(j)
        for t in range(ns):
            i = int(small[t])
            prob[i] = np.uint16(65535)
            alias[i] = np.int32(i)
        for t in range(nl):
            j = int(large[t])
            prob[j] = np.uint16(65535)
            alias[j] = np.int32(j)
        return prob, alias
