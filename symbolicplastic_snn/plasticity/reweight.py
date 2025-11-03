from __future__ import annotations

import numpy as np

from .update import reweight_alias_smallstep as _reweight_impl


def reweight_alias_smallstep(
    prob_q016: np.ndarray,
    corr_tile: np.ndarray,
    lr_num: int = 1,
    lr_den: int = 50,
    keep_sum: bool = True,
    ei_quota: tuple[float, float] | None = None,
    long_range_ratio: float | None = None,
) -> np.ndarray:
    """Macro-level integer small-step reweighting for alias probabilities.

    Delegates to the integer implementation in plasticity.update, preserving
    exact-sum normalization (Σ=65535) and optional E/I and long-range quotas.
    """
    prob = np.asarray(prob_q016, dtype=np.uint16)
    corr = np.asarray(corr_tile, dtype=np.int32)
    return _reweight_impl(
        prob_q016=prob,
        corr_int32=corr,
        lr_num=int(lr_num),
        lr_den=int(lr_den),
        keep_sum=bool(keep_sum),
        ei_quota=ei_quota,
        long_range_ratio=long_range_ratio,
    )


__all__ = ["reweight_alias_smallstep"]

