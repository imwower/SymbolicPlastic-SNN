from __future__ import annotations

from typing import Dict

import numpy as np

from .heavy_hitters import SpaceSavingK


def update_edge_usage(
    pre_ids: np.ndarray,
    post_ids: np.ndarray,
    hh_maps: Dict[int, SpaceSavingK],
    *,
    capacity: int = 64,
) -> None:
    """Batch-update heavy-hitter maps for (pre, post) pairs.

    - pre_ids/post_ids: int arrays of same shape (1-D)
    - hh_maps: dict pre_id -> SpaceSavingK (created on demand)
    - capacity: default capacity when creating new SpaceSavingK
    """
    pre = np.asarray(pre_ids).astype(np.int64, copy=False)
    post = np.asarray(post_ids).astype(np.int64, copy=False)
    if pre.shape[0] != post.shape[0]:
        raise ValueError("pre_ids and post_ids must have same length")
    n = pre.shape[0]
    for i in range(n):
        p = int(pre[i])
        q = int(post[i])
        tracker = hh_maps.get(p)
        if tracker is None:
            tracker = SpaceSavingK(capacity=int(capacity))
            hh_maps[p] = tracker
        tracker.update(q)


__all__ = ["update_edge_usage"]

