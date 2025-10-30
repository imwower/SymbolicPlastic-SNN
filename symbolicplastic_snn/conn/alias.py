from __future__ import annotations

from typing import Callable, Tuple
from dataclasses import dataclass

import numpy as np


def _prefix_sum_uint16(a: np.ndarray) -> np.ndarray:
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
    total = int(_prefix_sum_uint16(prob_q016)[-1])

    prob = np.zeros(n, dtype=np.uint16)
    alias = np.arange(n, dtype=np.int32)

    if total == 0:
        # Uniform fallback: each column is always chosen when its slot is picked.
        prob.fill(np.uint16(65535))
        return prob, alias

    # Scale weights by n to compare against total mass (Vose's method).
    # Si = wi * n, compare with total.
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
            # Exactly equal to 1.0 in scaled domain.
            prob[i] = np.uint16(65535)
            alias[i] = np.int32(i)
            # Do not push to stacks.

    # Process pairs
    while small and large:
        i = small.pop()
        j = large.pop()

        # Threshold for i: floor((Si / total) * 2^16). Keep in [0, 65535].
        # Use 2^16 scaling then compare using r2 < threshold.
        thr = (np.int64(S[i]) << 16) // total64  # in [0, 65536)
        if thr >= 65536:
            thr = 65535
        prob[i] = np.uint16(thr)
        alias[i] = np.int32(j)

        # Reduce Sj by the leftover probability allocated to fill column i.
        # New Sj = Sj - (total - Si)
        Sj_new = S[j] - (total64 - S[i])
        S[j] = Sj_new
        if Sj_new < total64:
            small.append(j)
        elif Sj_new > total64:
            large.append(j)
        else:
            prob[j] = np.uint16(65535)
            alias[j] = np.int32(j)

    # Any remaining entries are exactly 1.0 columns.
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


__all__ = [
    "build_alias",
    "sample_alias",
    "reweight_alias",
    "AliasForTile",
]


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
