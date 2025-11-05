from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from symbolicplastic_snn.utils.prng import Stream


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


def rate_encode_q016(x_q: np.ndarray, rng_uint16: Callable[[int], np.ndarray]) -> np.ndarray:
    """Integer Bernoulli sampling with Q0.16 probabilities.

    - x_q: uint16 array where value v means p = v / 65535.
    - rng_uint16(size): returns uint16 uniform in [0, 65535].
    Returns boolean mask of same shape.
    """
    xq = np.asarray(x_q, dtype=np.uint16)
    r = np.asarray(rng_uint16(xq.size), dtype=np.uint16).reshape(xq.shape)
    return r < xq

__all__.extend(["rate_encode_q016"])


def constant_q16(value: float, scale: float) -> np.uint16:
    """Encode a constant float value into Q0.16 with scaling.

    result = clip(round(value * scale * 65535), 0, 65535) as int16

    Examples
    >>> constant_q16(0.5, 1.0)
    np.int16(32768)
    """
    v = float(value) * float(scale)
    q = int(round(max(0.0, min(1.0, v)) * 65536.0))
    if q < 0:
        q = 0
    if q > 65535:
        q = 65535
    return np.uint16(q)


def piecewise_linear(x: float, knots: np.ndarray, values: np.ndarray) -> np.uint16:
    """Piecewise-linear mapping of x to Q0.16 given knots and values.

    - knots: strictly increasing float array
    - values: same length, in [0,1]
    """
    xs = float(x)
    k = np.asarray(knots, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if k.ndim != 1 or v.ndim != 1 or k.size != v.size or k.size == 0:
        raise ValueError("knots and values must be 1-D and same non-zero length")
    if np.any(k[1:] <= k[:-1]):
        raise ValueError("knots must be strictly increasing")
    # Clamp to endpoints
    if xs <= k[0]:
        y = v[0]
    elif xs >= k[-1]:
        y = v[-1]
    else:
        idx = int(np.searchsorted(k, xs))
        x0, x1 = k[idx - 1], k[idx]
        y0, y1 = v[idx - 1], v[idx]
        t = (xs - x0) / (x1 - x0)
        y = (1.0 - t) * y0 + t * y1
    y = max(0.0, min(1.0, float(y)))
    return np.uint16(int(round(y * 65536.0)) if y < 1.0 else 65535)


def poisson_spikes(rate_hz: float, dt_ms: float, stream: Stream) -> bool:
    """Poisson spike decision using Stream.uniform().

    Returns True with probability p = min(rate_hz * dt_ms / 1000, 1.0).
    """
    if rate_hz < 0 or dt_ms <= 0:
        raise ValueError("rate_hz must be >= 0 and dt_ms > 0")
    p = rate_hz * (dt_ms / 1000.0)
    p = max(0.0, min(1.0, float(p)))
    return bool(stream.uniform() < p)


def ratio_norm(x: float, lo: float, hi: float) -> np.uint16:
    """Map x in [lo, hi] to Q0.16 linearly.

    Clips outside range. If hi == lo, returns 0.
    """
    lo2 = float(lo)
    hi2 = float(hi)
    if hi2 <= lo2:
        return np.int16(0)
    t = (float(x) - lo2) / (hi2 - lo2)
    t = max(0.0, min(1.0, t))
    return np.uint16(int(round(t * 65536.0)) if t < 1.0 else 65535)


__all__.extend(["constant_q16", "piecewise_linear", "poisson_spikes", "ratio_norm"])


# ---- High-level encoders (deterministic via SeedSpace/Stream) ----

class PoissonRateEncoder:
    """Bernoulli spike encoder from rates using deterministic Stream.

    - rate: array of shape (N,) or scalar; values are per-step probabilities in [0,1].
    - T: total steps to generate.
    - key_prefix: SeedSpace key prefix used to derive stream.
    - clip: optional (lo, hi) to clip rate before sampling.

    Example
    >>> from symbolicplastic_snn.core.prng import SeedSpace
    >>> enc = PoissonRateEncoder(rate=0.2, T=4)
    >>> out = enc.encode(SeedSpace(1234), trial=0)
    >>> out.shape
    (4, 1)
    """

    def __init__(self, rate: np.ndarray | float, T: int, key_prefix: str = "module=encode", clip: tuple[float, float] | None = None) -> None:
        self.T = int(T)
        self.key_prefix = str(key_prefix)
        self.clip = clip
        self._rate = rate

    def encode(self, seed_space, trial: int, tile: int | None = None) -> np.ndarray:
        from symbolicplastic_snn.core.prng import SeedSpace, Stream  # type: ignore

        if isinstance(self._rate, np.ndarray):
            r = np.asarray(self._rate, dtype=np.float32).reshape(-1)
        else:
            r = np.array([float(self._rate)], dtype=np.float32)
        if self.clip is not None:
            lo, hi = float(self.clip[0]), float(self.clip[1])
            r = np.clip(r, lo, hi)
        r = np.clip(r, 0.0, 1.0)

        keys = [self.key_prefix, f"trial={int(trial)}"]
        if tile is not None:
            keys.append(f"tile={int(tile)}")
        st = seed_space.derive(*keys)

        T, N = self.T, int(r.size)
        out = np.zeros((T, N), dtype=np.int8)
        # Draw T*N uniforms deterministically
        for t in range(T):
            for i in range(N):
                out[t, i] = 1 if st.uniform() < float(r[i]) else 0
        return out


class LatencyRankEncoder:
    """Latency encoder: earlier time for stronger intensity with deterministic tie-breaks.

    - window: number of time steps [1..]
    - key_prefix: SeedSpace key prefix used to derive stream for tie-breaking.

    Each neuron spikes exactly once within the `window`. If N > window,
    ranks wrap around with modulo (rank % window).

    Example
    >>> from symbolicplastic_snn.core.prng import SeedSpace
    >>> enc = LatencyRankEncoder(window=4)
    >>> x = np.array([0.8, 0.1, 0.8], dtype=np.float32)
    >>> out = enc.encode(x, SeedSpace(0), trial=0)
    >>> out.sum(axis=0).tolist()
    [1, 1, 1]
    """

    def __init__(self, window: int, key_prefix: str = "module=encode") -> None:
        self.window = max(1, int(window))
        self.key_prefix = str(key_prefix)

    def encode(self, intensities: np.ndarray, seed_space, trial: int) -> np.ndarray:
        x = np.asarray(intensities, dtype=np.float32).reshape(-1)
        N = int(x.size)

        # Group indices by intensity value for stable tie-breaks
        vals = {}
        for i, v in enumerate(x.tolist()):
            vals.setdefault(float(v), []).append(i)

        # Derive tie-break RNG once per trial
        st = seed_space.derive(self.key_prefix, f"trial={int(trial)}")

        order: list[int] = []
        # Sort intensities descending, then shuffle equal-value groups
        for v in sorted(vals.keys(), reverse=True):
            grp = vals[v]
            # Deterministic shuffle via Stream
            a = np.array(grp, dtype=np.int64)
            st.shuffle(a)
            order.extend(a.astype(int).tolist())

        # Map rank to time t in [0, window-1] with wrap-around
        out = np.zeros((self.window, N), dtype=np.int8)
        for rank, idx in enumerate(order):
            t = rank % self.window
            out[t, int(idx)] = 1
        return out
