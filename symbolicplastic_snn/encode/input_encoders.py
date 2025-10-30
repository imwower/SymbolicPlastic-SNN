from __future__ import annotations

from typing import Tuple

import numpy as np


def rate_encode(x: np.ndarray, rate_max: float, rng) -> np.ndarray:
    """Simple rate encoding: Bernoulli spikes with p = clip(x * rate_max, 0, 1).

    - x: float array
    - rng: object with .random(size) -> floats in [0, 1)
    Returns bool mask of same shape as x.
    """
    xx = np.asarray(x, dtype=np.float32)
    p = np.clip(xx * float(rate_max), 0.0, 1.0)
    r = rng.random(xx.shape)
    return r < p


def latency_encode(x: np.ndarray, window: int) -> np.ndarray:
    """Latency encoding schedule across `window` steps.

    For each input value in [0, 1], produce a one-hot spike time where
    earlier spikes correspond to larger input values. Returns a boolean
    array of shape (len(x), window).
    """
    xx = np.asarray(x, dtype=np.float32)
    n = xx.shape[0]
    w = int(window)
    w = max(1, w)
    # Map x in [0,1] to time index t in [0, w-1] using t = floor((1-x)*(w-1))
    t = np.floor((1.0 - np.clip(xx, 0.0, 1.0)) * (w - 1)).astype(int)
    out = np.zeros((n, w), dtype=bool)
    out[np.arange(n), t] = True
    return out


def diff_encode(x_t: np.ndarray, x_tm1: np.ndarray, thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Differential encoding: spike on positive/negative changes above threshold.

    Returns (pos_mask, neg_mask) as boolean arrays.
    """
    a = np.asarray(x_t, dtype=np.float32)
    b = np.asarray(x_tm1, dtype=np.float32)
    d = a - b
    pos = d >= float(thr)
    neg = d <= -float(thr)
    return pos, neg


__all__ = ["rate_encode", "latency_encode", "diff_encode"]

