from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ConfigFp:
    """Fixed-point configuration for LIF kernel.

    Attributes
    - refractory_steps: Refractory duration in steps after a spike.
    - q_v_theta: Q format for v/theta, default "Q4.11" (int16).
    - lambda_q15: Whether leak uses Q1.15 multiplier (True) or shift approx.
    - v_reset: Reset potential used during refractory (default 0).

    Notes
    - All updates are deterministic and vectorized (no Python loops).
    - This config does not change function semantics beyond controlling
      leak and refractory behavior.
    """

    refractory_steps: int = 6
    q_v_theta: str = "Q4.11"
    lambda_q15: bool = True
    v_reset: int = 0


def saturating_int16(x: np.ndarray) -> np.ndarray:
    """Saturate an int32 array to int16 range [-32768, 32767].

    Parameters
    - x: int32 ndarray.

    Returns
    - int16 ndarray with saturation.

    Examples
    >>> import numpy as np
    >>> from symbolicplastic_snn.core.lif_fixedpoint import saturating_int16
    >>> arr = np.array([40000, -50000, 123], dtype=np.int32)
    >>> saturating_int16(arr)
    array([ 32767, -32768,    123], dtype=int16)
    """

    if x.dtype != np.int32 and x.dtype != np.int64:
        raise TypeError("x must be int32/int64 ndarray for saturation")
    x64 = x.astype(np.int64, copy=False)
    x64 = np.clip(x64, -32768, 32767)
    return x64.astype(np.int16, copy=False)


def q15_mul_round(a: np.ndarray, lam: np.ndarray | int) -> np.ndarray:
    """Multiply int16 `a` by Q1.15 `lam` with rounding, return int16.

    Computes round((a * lam) / 2^15) using symmetric rounding.
    For positive products: add 2^14; for negative: add 2^14 - 1 before shift.

    Parameters
    - a: int16 ndarray.
    - lam: uint16 scalar or ndarray (Q1.15).

    Returns
    - int16 ndarray.

    Examples
    >>> import numpy as np
    >>> from symbolicplastic_snn.core.lif_fixedpoint import q15_mul_round
    >>> a = np.array([1000, -1000], dtype=np.int16)
    >>> lam = np.uint16(16384)  # ~0.5 in Q1.15
    >>> q15_mul_round(a, lam)
    array([ 500, -500], dtype=int16)
    """

    if a.dtype != np.int16:
        raise TypeError("a must be int16 ndarray")
    # Ensure broadcasting and integer precision during intermediate ops.
    prod = a.astype(np.int32) * np.asarray(lam, dtype=np.uint16).astype(np.int32)
    # Round to nearest (ties away from zero) using bias depending on sign
    bias = np.where(prod >= 0, 1 << 14, (1 << 14) - 1).astype(np.int32)
    shifted = (prod + bias) >> 15
    # Saturate to int16 for safety (though range usually fits)
    return saturating_int16(shifted.astype(np.int32))


def lif_step(
    v: np.ndarray,
    ref: np.ndarray,
    I: np.ndarray,
    theta: np.int16,
    lambda_q15: np.uint16,
    cfg: ConfigFp,
) -> Tuple[np.ndarray, int]:
    """One vectorized fixed-point LIF update step.

    Behavior
    - Indices with ref > 0: ref -= 1; v set to cfg.v_reset (no integration).
    - Others: vv = round((lambda_q15 * v) / 2^15) + I; saturate to int16.
      Spike if vv >= theta: spike=True, v=0, ref=cfg.refractory_steps.

    Parameters
    - v: int16 ndarray (membrane potential, Q4.11 by default).
    - ref: uint8 ndarray (refractory counters).
    - I: int32 ndarray (event/current accumulator, same Q domain as v).
    - theta: int16 scalar threshold (same Q as v).
    - lambda_q15: uint16 scalar (Q1.15 leak multiplier).
    - cfg: ConfigFp.

    Returns
    - spike_mask (bool ndarray), num_spikes (int).

    Notes
    - Operates in-place on `v` and `ref` for performance.
    - Deterministic: no randomness; pure NumPy vectorization.

    Examples
    >>> import numpy as np
    >>> from symbolicplastic_snn.core.lif_fixedpoint import ConfigFp, lif_step
    >>> v = np.array([0, 1000], dtype=np.int16)
    >>> ref = np.array([0, 0], dtype=np.uint8)
    >>> I = np.array([0, 40000], dtype=np.int32)
    >>> theta = np.int16(20000)
    >>> lam = np.uint16(32768)  # ~1.0
    >>> cfg = ConfigFp(refractory_steps=3)
    >>> spikes, n = lif_step(v, ref, I, theta, lam, cfg)
    >>> n, spikes.tolist(), v.tolist(), ref.tolist()
    (1, [False, True], [0, 0], [0, 3])
    """

    # Type checks for deterministic, predictable behavior.
    if v.dtype != np.int16:
        raise TypeError("v must be int16 ndarray")
    if ref.dtype != np.uint8:
        raise TypeError("ref must be uint8 ndarray")
    if I.dtype != np.int32:
        raise TypeError("I must be int32 ndarray")
    if not isinstance(theta, (np.int16, np.int32, int)):
        raise TypeError("theta must be int16-compatible scalar")
    if not isinstance(lambda_q15, (np.uint16, np.uint32, int)):
        raise TypeError("lambda_q15 must be uint16-compatible scalar")

    theta_val = np.int16(theta).item()
    lam_val = np.uint16(lambda_q15).item()

    n = v.shape[0]
    if ref.shape[0] != n or I.shape[0] != n:
        raise ValueError("v, ref, I must have the same length")

    # Masks
    ref_mask = ref > 0
    active_mask = ~ref_mask

    # Refractory: decrement and clamp v to reset (no integration)
    if np.any(ref_mask):
        ref[ref_mask] = (ref[ref_mask] - 1).astype(np.uint8, copy=False)
        v[ref_mask] = np.int16(cfg.v_reset)

    # Active indices: integrate leak and input, saturate to int16
    if np.any(active_mask):
        v_act = v[active_mask]
        I_act = I[active_mask]

        if cfg.lambda_q15:
            leak = q15_mul_round(v_act, np.uint16(lam_val)).astype(np.int32)
        else:
            # Shift approximation: v - (v >> p) with p chosen from lam; not specified
            # Keep deterministic fallback to q15 path for now (lambda_q15=True expected)
            leak = q15_mul_round(v_act, np.uint16(lam_val)).astype(np.int32)

        vv32 = leak + I_act
        vv16 = saturating_int16(vv32)

        # Spikes for active indices only
        spikes_act = vv16.astype(np.int32) >= np.int32(theta_val)

        # Write back potentials
        v[active_mask] = np.where(spikes_act, np.int16(0), vv16)

        # Update ref for spikes
        if np.any(spikes_act):
            # Set refractory steps at spiking indices within the active subset
            ref_idx = np.nonzero(active_mask)[0]
            ref_spk_idx = ref_idx[spikes_act]
            ref[ref_spk_idx] = np.uint8(cfg.refractory_steps)

        spike_mask = np.zeros(n, dtype=bool)
        spike_mask[active_mask] = spikes_act
    else:
        spike_mask = np.zeros(n, dtype=bool)

    num_spikes = int(spike_mask.sum())
    return spike_mask, num_spikes


__all__ = [
    "ConfigFp",
    "saturating_int16",
    "q15_mul_round",
    "lif_step",
]
