from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from symbolicplastic_snn.plasticity.reweight import reweight_alias_smallstep
from symbolicplastic_snn.plasticity.reseed import reseed_small_fraction
from symbolicplastic_snn.plasticity.consolidation import (
    PromoteConfig,
    promote_candidates,
    demote_candidates,
    flip_sign_updates,
)
from symbolicplastic_snn.plasticity.stable_store import StableStore
from symbolicplastic_snn.plasticity.heavy_hitters import SpaceSavingK


def apply_plasticity_pipeline(
    prob_q016: np.ndarray,
    corr_tile: np.ndarray,
    seeds_flex: np.ndarray,
    reseed_mask: np.ndarray,
    store: StableStore,
    hh_maps: Dict[int, SpaceSavingK],
    corr_map: Dict[Tuple[int, int], int],
    ages: Dict[Tuple[int, int], int],
    *,
    lr_num: int = 1,
    lr_den: int = 50,
    ei_quota: tuple[float, float] | None = None,
    long_range_ratio: float | None = None,
    reseed_rate: float = 0.01,
    epoch: int = 0,
    promote_cfg: PromoteConfig | None = None,
) -> Dict[str, int | np.ndarray]:
    """Run macro reweight, micro reseed, and consolidation in-order.

    Returns dict with updated prob (`prob`), counts for reseeded/promoted/demoted/flipped.
    """
    pcfg = promote_cfg or PromoteConfig()

    # 1) Macro: alias reweight
    new_prob = reweight_alias_smallstep(
        prob_q016=np.asarray(prob_q016, dtype=np.uint16),
        corr_tile=np.asarray(corr_tile, dtype=np.int32),
        lr_num=lr_num,
        lr_den=lr_den,
        keep_sum=True,
        ei_quota=ei_quota,
        long_range_ratio=long_range_ratio,
    )

    # 2) Micro: reseed flex-only for exploration
    reseeded = reseed_small_fraction(seeds_flex=np.asarray(seeds_flex, dtype=np.uint64), mask=np.asarray(reseed_mask, dtype=bool), rate=reseed_rate, epoch=epoch)

    # 3) Consolidation: promote/demote/flip
    promoted = 0
    for pre_id, hh in hh_maps.items():
        added = promote_candidates(int(pre_id), hh, corr_map, ages, store, pcfg)
        promoted += len(added)
    removed = demote_candidates(store, corr_map, ages, pcfg)
    demoted = len(removed)
    flipped = flip_sign_updates(store, corr_map, pcfg)

    return {
        "prob": new_prob,
        "reseeded": int(reseeded),
        "promoted": int(promoted),
        "demoted": int(demoted),
        "flipped": int(flipped),
    }


__all__ = ["apply_plasticity_pipeline"]

