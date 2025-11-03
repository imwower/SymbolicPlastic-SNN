from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable

import numpy as np

from .heavy_hitters import SpaceSavingK
from .stable_store import StableStore, StableEdge
from .states import EDGE_STABLE, EDGE_CONS


@dataclass
class PromoteConfig:
    min_age: int = 5_000
    min_corr: int = 10
    min_hits: int = 50
    per_pre_cap: int = 128
    demote_age: int = 8_000
    demote_corr: int = -10
    flip_sign_pos: int = 20
    flip_sign_neg: int = -20


def promote_candidates(
    pre_id: int,
    hh: SpaceSavingK,
    corr_map: Dict[Tuple[int, int], int],
    ages: Dict[Tuple[int, int], int],
    store: StableStore,
    cfg: PromoteConfig,
) -> List[StableEdge]:
    """Select and insert stable edges based on heavy-hitters + corr + age.

    - For items in hh.topk(), if count>=min_hits and corr>=min_corr and age>=min_age,
      and not already in store, insert until per-pre capacity is reached.
    - Delay defaults to 0 for promoted edges; sign inferred by corr thresholds.
    Returns list of edges actually inserted.
    """
    added: List[StableEdge] = []
    # Count current per-pre entries to respect cap
    cur = store.get_pre(int(pre_id))
    remaining = max(0, int(cfg.per_pre_cap) - len(cur))
    if remaining <= 0:
        return added

    for (post_id, cnt, err) in hh.topk():
        if remaining <= 0:
            break
        c = int(corr_map.get((int(pre_id), int(post_id)), 0))
        a = int(ages.get((int(pre_id), int(post_id)), 0))
        if cnt >= int(cfg.min_hits) and c >= int(cfg.min_corr) and a >= int(cfg.min_age):
            # Skip if exists for any delay 0 (our default)
            if store.exists(int(pre_id), int(post_id), 0):
                continue
            # Determine sign by flip thresholds; default +1
            sg = 1
            if c >= int(cfg.flip_sign_pos):
                sg = 1
            elif c <= int(cfg.flip_sign_neg):
                sg = -1
            e = StableEdge(
                pre_id=int(pre_id),
                post_id=int(post_id),
                sign=np.int8(sg),
                delay=np.uint8(0),
                state=np.uint8(EDGE_STABLE),
                age=np.uint16(min(a, 0xFFFF)),
                corr=np.int16(max(-32768, min(32767, c))),
                last_used=np.int32(0),
            )
            if store.add(e):
                added.append(e)
                remaining -= 1
    return added


def demote_candidates(
    store: StableStore,
    corr_map: Dict[Tuple[int, int], int],
    ages: Dict[Tuple[int, int], int],
    cfg: PromoteConfig,
) -> List[StableEdge]:
    """Remove edges that are idle too long or have strongly negative corr.

    - Edges with state==CONS are protected and skipped.
    - Demotion criteria (OR): age>=demote_age or corr<=demote_corr.
    Returns list of edges removed.
    """
    removed: List[StableEdge] = []
    # Collect keys to remove first (can't modify while iterating store map)
    to_remove: List[Tuple[int, int, int]] = []
    for e in store.iter_all():
        if int(e.state) == EDGE_CONS:
            continue
        a = int(ages.get((int(e.pre_id), int(e.post_id)), int(e.age)))
        c = int(corr_map.get((int(e.pre_id), int(e.post_id)), int(e.corr)))
        if a >= int(cfg.demote_age) or c <= int(cfg.demote_corr):
            to_remove.append((int(e.pre_id), int(e.post_id), int(e.delay)))
    for pid, qid, dly in to_remove:
        # Retrieve before delete for return
        # Reconstruct edge info for reporting
        edges = store.get_pre(pid)
        target = None
        for e in edges:
            if int(e.post_id) == qid and int(e.delay) == dly:
                target = e
                break
        if store.remove(pid, qid, dly) and target is not None:
            removed.append(target)
    return removed


def flip_sign_updates(store: StableStore, corr_map: Dict[Tuple[int, int], int], cfg: PromoteConfig) -> int:
    """Flip signs of stable edges based on corr thresholds.

    Returns number of edges updated.
    """
    flips = 0
    for e in store.iter_all():
        c = int(corr_map.get((int(e.pre_id), int(e.post_id)), int(e.corr)))
        prev = int(e.sign)
        if c >= int(cfg.flip_sign_pos):
            e.sign = np.int8(1)
        elif c <= int(cfg.flip_sign_neg):
            e.sign = np.int8(-1)
        if int(e.sign) != prev:
            flips += 1
    return flips


def freeze_and_unfreeze(store: StableStore, selector: Callable[[StableEdge], bool], action: str) -> int:
    """Freeze (CONS) or unfreeze (STABLE) edges matching selector.

    Returns number of edges changed.
    """
    act = action.lower().strip()
    changed = 0
    for e in store.iter_all():
        if not selector(e):
            continue
        if act == "freeze":
            if int(e.state) != EDGE_CONS:
                e.state = np.uint8(EDGE_CONS)
                changed += 1
        elif act == "unfreeze":
            if int(e.state) != EDGE_STABLE:
                e.state = np.uint8(EDGE_STABLE)
                changed += 1
        else:
            raise ValueError("action must be 'freeze' or 'unfreeze'")
    return changed


__all__ = [
    "PromoteConfig",
    "promote_candidates",
    "demote_candidates",
    "flip_sign_updates",
    "freeze_and_unfreeze",
]

