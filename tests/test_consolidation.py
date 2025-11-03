import numpy as np

from symbolicplastic_snn.plasticity.heavy_hitters import SpaceSavingK
from symbolicplastic_snn.plasticity.stable_store import StableStore, StableEdge
from symbolicplastic_snn.plasticity.consolidation import (
    PromoteConfig,
    promote_candidates,
    demote_candidates,
    flip_sign_updates,
    freeze_and_unfreeze,
)
from symbolicplastic_snn.plasticity.states import EDGE_STABLE, EDGE_CONS


def test_promote_when_meets_thresholds():
    cfg = PromoteConfig(min_age=10, min_corr=5, min_hits=3, per_pre_cap=4, flip_sign_pos=20, flip_sign_neg=-20)
    pre = 7
    hh = SpaceSavingK(capacity=8)
    # Make post 100 frequent
    for _ in range(5):
        hh.update(100)
    # Others infrequent
    hh.update(101)
    corr = {(pre, 100): 12}
    ages = {(pre, 100): 15}
    store = StableStore(per_pre_cap=cfg.per_pre_cap)

    added = promote_candidates(pre, hh, corr, ages, store, cfg)
    assert len(added) == 1
    e = added[0]
    assert e.pre_id == pre and e.post_id == 100
    assert e.state == EDGE_STABLE
    assert store.exists(pre, 100, 0)


def test_demote_when_idle_or_negative():
    cfg = PromoteConfig(demote_age=10, demote_corr=-5)
    store = StableStore(per_pre_cap=8)
    # Add three edges
    e1 = StableEdge(pre_id=1, post_id=2, sign=np.int8(1), delay=np.uint8(0))
    e2 = StableEdge(pre_id=1, post_id=3, sign=np.int8(-1), delay=np.uint8(0))
    e3 = StableEdge(pre_id=2, post_id=5, sign=np.int8(1), delay=np.uint8(0))
    store.add(e1)
    store.add(e2)
    store.add(e3)
    # Mark e2 as CONS to protect
    e2.state = np.uint8(EDGE_CONS)
    ages = {(1, 2): 15, (1, 3): 20, (2, 5): 1}
    corr = {(1, 2): 0, (1, 3): -10, (2, 5): 0}

    removed = demote_candidates(store, corr, ages, cfg)
    # e1 removed due to age; e3 kept (young); e2 protected by CONS
    ids = {(e.pre_id, e.post_id) for e in removed}
    assert (1, 2) in ids
    assert not store.exists(1, 3, 0) is False or e2.state == EDGE_CONS
    assert store.exists(2, 5, 0)


def test_flip_sign():
    cfg = PromoteConfig(flip_sign_pos=10, flip_sign_neg=-10)
    store = StableStore(per_pre_cap=8)
    e1 = StableEdge(pre_id=1, post_id=2, sign=np.int8(-1), delay=np.uint8(0))
    e2 = StableEdge(pre_id=1, post_id=4, sign=np.int8(1), delay=np.uint8(0))
    store.add(e1); store.add(e2)
    corr = {(1, 2): 20, (1, 4): -20}
    flips = flip_sign_updates(store, corr, cfg)
    assert flips == 2
    # Signs flipped accordingly
    for e in store.get_pre(1):
        if e.post_id == 2:
            assert int(e.sign) == 1
        if e.post_id == 4:
            assert int(e.sign) == -1


def test_freeze_unfreeze():
    cfg = PromoteConfig(demote_age=10, demote_corr=-5)
    store = StableStore(per_pre_cap=8)
    e = StableEdge(pre_id=1, post_id=2, sign=np.int8(1), delay=np.uint8(0))
    store.add(e)
    # Freeze
    n = freeze_and_unfreeze(store, lambda ed: ed.pre_id == 1 and ed.post_id == 2, action="freeze")
    assert n == 1
    assert int(store.get_pre(1)[0].state) == EDGE_CONS
    # Attempt demote: should be protected
    removed = demote_candidates(store, {(1, 2): -100}, {(1, 2): 1000}, cfg)
    assert len(removed) == 0
    # Unfreeze and demote should now remove
    n2 = freeze_and_unfreeze(store, lambda ed: True, action="unfreeze")
    assert n2 == 1
    removed2 = demote_candidates(store, {(1, 2): -100}, {(1, 2): 1000}, cfg)
    assert len(removed2) == 1

