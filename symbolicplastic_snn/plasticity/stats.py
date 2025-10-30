from __future__ import annotations

from dataclasses import dataclass
import numpy as np


_MASK32 = np.uint32(0xFFFFFFFF)


@dataclass
class BitWindow:
    """Compact per-neuron spike history as a fixed-width bit window.

    - width: number of bits retained (<= 32). Newest bit at LSB (bit 0).
    - push(spike_mask): shifts left by 1 and inserts current spikes at bit 0.
    - get_history(idx): returns uint32 history word for neuron idx.
    """

    width: int = 32

    def __post_init__(self) -> None:
        if not (1 <= int(self.width) <= 32):
            raise ValueError("width must be in [1, 32]")
        self._mask = np.uint32((1 << int(self.width)) - 1)
        self._hist: np.ndarray | None = None  # uint32 per neuron

    def push(self, spike_mask: np.ndarray) -> None:
        sm = np.asarray(spike_mask, dtype=bool)
        n = sm.size
        if self._hist is None:
            self._hist = np.zeros(n, dtype=np.uint32)
        elif self._hist.size != n:
            raise ValueError("spike_mask length mismatch with initialized history")

        # Shift left and insert current bit at LSB.
        self._hist = ((self._hist << np.uint32(1)) | sm.astype(np.uint32)) & self._mask

    def get_history(self, idx: int) -> np.uint32:
        if self._hist is None:
            raise ValueError("history is empty; push() has not been called")
        return np.uint32(self._hist[int(idx)])


def _shift_u32(x: int | np.uint32, delta: int) -> int:
    v = int(np.uint32(x))
    if delta >= 0:
        return (v << delta) & 0xFFFFFFFF
    else:
        return (v >> (-delta)) & 0xFFFFFFFF


def update_corr(
    pre_hist: np.uint32 | int,
    post_hist: np.uint32 | int,
    delta: int,
    alpha_pos: np.uint8 | int = np.uint8(2),
    alpha_neg: np.uint8 | int = np.uint8(1),
) -> np.int32:
    """Integer STDP correlation update using bit histories.

    Computes: corr += popcount(pre & (post << delta)) * alpha_pos
                    - popcount(post & (pre << delta)) * alpha_neg

    Returns the signed int32 delta to be accumulated by the caller.
    """
    pre = int(np.uint32(pre_hist))
    post = int(np.uint32(post_hist))
    sh_post = _shift_u32(post, int(delta))
    sh_pre = _shift_u32(pre, int(delta))

    pos = (pre & sh_post)
    neg = (post & sh_pre)
    c_pos = int(pos.bit_count())
    c_neg = int(neg.bit_count())

    a_pos = int(np.uint8(alpha_pos))
    a_neg = int(np.uint8(alpha_neg))
    out = c_pos * a_pos - c_neg * a_neg
    return np.int32(out)


__all__ = ["BitWindow", "update_corr"]

