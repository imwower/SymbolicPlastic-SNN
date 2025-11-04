from __future__ import annotations

import numpy as np


def clamp16(x: int) -> np.int16:
    """Clamp Python int to int16 range [-32768, 32767] and return np.int16.

    Useful when converting intermediate integer results back to Q formats.
    """
    if x > 32767:
        x = 32767
    elif x < -32768:
        x = -32768
    return np.int16(x)


def q_add_sat(a: np.int16, b: np.int16) -> np.int16:
    """Saturating add for int16 operands.

    Returns a + b clamped to int16 bounds. Inputs are treated as signed 16-bit.
    """
    ai = int(np.int16(a))
    bi = int(np.int16(b))
    return clamp16(ai + bi)


def q_sub_sat(a: np.int16, b: np.int16) -> np.int16:
    """Saturating subtraction for int16 operands.

    Returns a - b clamped to int16 bounds.
    """
    ai = int(np.int16(a))
    bi = int(np.int16(b))
    return clamp16(ai - bi)


def q_mul_q(a: np.int16, b: np.int16, q_frac_bits: int) -> np.int16:
    """Fixed-point multiply for two signed Q-format int16 values with saturation.

    - a, b: int16 operands representing Qx.q_frac_bits values.
    - q_frac_bits: number of fractional bits in the representation (e.g., 11 for Q4.11).
    - Returns: (a * b) >> q_frac_bits, saturated to int16 range.
    """
    ai = int(np.int16(a))
    bi = int(np.int16(b))
    prod = ai * bi  # up to 31-bit magnitude
    # Arithmetic shift (Python // handles sign; use right shift on int for power-of-two)
    if q_frac_bits >= 0:
        res = prod >> int(q_frac_bits)
    else:
        res = prod << int(-q_frac_bits)
    return clamp16(res)


__all__ = ["q_add_sat", "q_sub_sat", "q_mul_q", "clamp16"]

