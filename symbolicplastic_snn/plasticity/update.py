from __future__ import annotations

from typing import Tuple

import numpy as np


def _normalize_to_sum_uint16(x: np.ndarray, target_sum: int = 65535) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    total = int(x.sum())
    n = x.size
    if total <= 0 or n == 0:
        if n == 0:
            return np.zeros(0, dtype=np.uint16)
        base = target_sum // n
        out = np.full(n, base, dtype=np.int32)
        remainder = target_sum - base * n
        if remainder > 0:
            out[:remainder] += 1
        return out.astype(np.uint16)
    prod = x * int(target_sum)
    base = (prod // total).astype(np.int32)
    rem = (prod % total).astype(np.int64)
    out = base.copy()
    need = int(target_sum - int(base.sum(dtype=np.int64)))
    if need > 0:
        if need < n:
            idx = np.argpartition(-rem, need - 1)[:need]
        else:
            idx = np.arange(n)
        sel = idx[np.argsort(-rem[idx], kind="mergesort")][:need]
        out[sel] += 1
    return out.astype(np.uint16)


def _rescale_group_int(x: np.ndarray, target: int) -> np.ndarray:
    """Scale a non-negative int array so its sum becomes `target` (int), deterministically.

    Uses proportional scaling with integer rounding and remainder distribution by largest remainders.
    If the group sum is zero, distribute evenly to match target.
    """
    g = np.asarray(x, dtype=np.int64)
    n = g.size
    target = max(0, int(target))
    s = int(g.sum())
    if n == 0:
        return g.astype(np.int32)
    if s == 0:
        base = target // n
        out = np.full(n, base, dtype=np.int32)
        rem = target - base * n
        if rem > 0:
            out[:rem] += 1
        return out
    prod = g * target
    base = (prod // s).astype(np.int32)
    rem = (prod % s).astype(np.int64)
    out = base.copy()
    need = int(target - int(base.sum(dtype=np.int64)))
    if need > 0:
        if need < n:
            idx = np.argpartition(-rem, need - 1)[:need]
        else:
            idx = np.arange(n)
        sel = idx[np.argsort(-rem[idx], kind="mergesort")][:need]
        out[sel] += 1
    return out


def reweight_alias_smallstep(
    prob_q016: np.ndarray,
    corr_int32: np.ndarray,
    lr_num: int,
    lr_den: int,
    keep_sum: bool = True,
    ei_quota: Tuple[float, float] | None = None,
    long_range_ratio: float | None = None,
) -> np.ndarray:
    """Integerized small-step alias reweighting with E/I and long-range quotas.

    Steps
    - prob += (corr * lr_num) // lr_den
    - Clamp to [0, 65535]
    - Project onto E/I quotas and long-range cap (deterministic integer scaling)
    - Normalize back to sum=65535 if keep_sum

    Assumptions
    - E/I partition: first half is E, second half is I.
    - Long-range set: last quarter of entries.
    """
    if prob_q016.dtype != np.uint16:
        raise TypeError("prob_q016 must be uint16 ndarray")
    if corr_int32.dtype not in (np.int32, np.int64):
        raise TypeError("corr_int32 must be int32/int64 ndarray")
    if prob_q016.shape[0] != corr_int32.shape[0]:
        raise ValueError("prob_q016 and corr_int32 must have same length")
    if lr_den == 0:
        raise ValueError("lr_den must be non-zero")

    n = prob_q016.shape[0]
    # Small-step update
    delta = (corr_int32.astype(np.int64) * int(lr_num)) // int(lr_den)
    x = prob_q016.astype(np.int64) + delta
    x = np.clip(x, 0, 65535).astype(np.int32)

    total = int(x.sum())
    if total == 0:
        out = np.zeros_like(x)
    else:
        out = x.copy()

    # E/I quota projection
    if ei_quota is not None and n > 0:
        qe, qi = float(ei_quota[0]), float(ei_quota[1])
        qe = max(0.0, min(1.0, qe))
        qi = max(0.0, min(1.0, qi))
        s = int(out.sum(dtype=np.int64))
        # Partition indices
        mid = n // 2
        E = slice(0, mid)
        I = slice(mid, n)
        target_E = int(qe * s)
        target_I = max(0, s - target_E) if (qe + qi) >= 1e-9 else s
        out_E = _rescale_group_int(out[E], target_E)
        out_I = _rescale_group_int(out[I], target_I)
        out = np.concatenate([out_E, out_I]).astype(np.int32)

    # Long-range cap projection
    if long_range_ratio is not None and n > 0:
        lr = float(long_range_ratio)
        lr = max(0.0, min(1.0, lr))
        s = int(out.sum(dtype=np.int64))
        if s > 0:
            # Last quarter indices considered long-range
            start = n - max(1, n // 4)
            LR = slice(start, n)
            NR = slice(0, start)
            cap = int(lr * s)
            s_lr = int(out[LR].sum(dtype=np.int64))
            if s_lr > cap:
                out_lr = _rescale_group_int(out[LR], cap)
                out_nr = _rescale_group_int(out[NR], s - cap)
                out = np.concatenate([out_nr, out_lr]).astype(np.int32)

    if keep_sum:
        return _normalize_to_sum_uint16(out, target_sum=65535)
    return np.clip(out, 0, 65535).astype(np.uint16)


__all__ = ["reweight_alias_smallstep"]

