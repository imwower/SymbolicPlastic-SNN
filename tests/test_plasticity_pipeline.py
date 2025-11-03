import numpy as np

from symbolicplastic_snn.plasticity.reweight import reweight_alias_smallstep
from symbolicplastic_snn.plasticity.reseed import reseed_small_fraction
from symbolicplastic_snn.runner.hooks import apply_plasticity_pipeline
from symbolicplastic_snn.plasticity.stable_store import StableStore
from symbolicplastic_snn.plasticity.heavy_hitters import SpaceSavingK
from symbolicplastic_snn.plasticity.consolidation import PromoteConfig


def test_alias_reweight_sum_and_quotas():
    n = 8
    # Uniform base over n tiles
    base = 65535 // n
    prob = np.full(n, base, dtype=np.uint16)
    rem = 65535 - base * n
    if rem > 0:
        prob[:rem] = (prob[:rem].astype(np.uint32) + 1).astype(np.uint16)
    # Corr: favor first half, discourage last quarter (as long-range cup)
    corr = np.array([5, 5, 4, 3, -2, -2, -3, -4], dtype=np.int32)
    out = reweight_alias_smallstep(prob, corr, lr_num=1, lr_den=10, keep_sum=True, ei_quota=(0.7, 0.3), long_range_ratio=0.25)
    assert out.dtype == np.uint16
    assert int(out.astype(np.uint32).sum()) == 65535
    # E/I check
    mid = n // 2
    sum_E = int(out[:mid].astype(np.uint32).sum())
    sum_I = int(out[mid:].astype(np.uint32).sum())
    target_E = int(0.7 * 65535)
    assert abs(sum_E - target_E) <= 1
    # Long-range = last quarter
    lr_start = n - max(1, n // 4)
    sum_LR = int(out[lr_start:].astype(np.uint32).sum())
    assert sum_LR <= int(0.25 * 65535) + 1


def test_reseed_only_flex_channel():
    n = 1024
    seeds_core = np.arange(n, dtype=np.uint64) * np.uint64(0xD1342543)
    seeds_flex = np.arange(n, dtype=np.uint64) * np.uint64(0x94D049BB)
    before_core = seeds_core.copy()
    before_flex = seeds_flex.copy()
    mask = np.zeros(n, dtype=bool)
    mask[::3] = True
    reseeded = reseed_small_fraction(seeds_flex, mask, rate=0.3, epoch=123)
    assert reseeded > 0
    # core unchanged
    assert np.array_equal(seeds_core, before_core)
    # flex changed only at masked positions (subset may change depending on rate)
    changed = seeds_flex != before_flex
    assert np.any(changed[mask])
    assert not np.any(changed[~mask])


def test_pipeline_order_effects():
    # Alias prob
    ntiles = 6
    base = 65535 // ntiles
    prob = np.full(ntiles, base, dtype=np.uint16)
    prob[: 65535 - base * ntiles] += 1
    # Corr favors early tiles
    corr_tile = np.array([5, 4, 3, -1, -2, -3], dtype=np.int32)
    # Seeds
    seeds_flex = np.arange(100, 100 + ntiles, dtype=np.uint64)
    reseed_mask = np.ones(ntiles, dtype=bool)
    # Stable store initially with one negative-corr edge
    store = StableStore(per_pre_cap=8)
    from symbolicplastic_snn.plasticity.stable_store import StableEdge
    store.add(StableEdge(pre_id=1, post_id=5, sign=np.int8(1), delay=np.uint8(0)))
    # corr_map makes this edge negative
    corr_map = {(1, 5): -100}
    ages = {(1, 5): 20_000}
    # Heavy-hitter: pre=2 frequently connects to post=3
    hh_maps = {2: SpaceSavingK(capacity=8)}
    for _ in range(10):
        hh_maps[2].update(3)
    corr_map[(2, 3)] = 50
    ages[(2, 3)] = 10_000

    pcfg = PromoteConfig(min_age=100, min_corr=10, min_hits=5, demote_age=1000, demote_corr=-10)
    out = apply_plasticity_pipeline(
        prob_q016=prob,
        corr_tile=corr_tile,
        seeds_flex=seeds_flex,
        reseed_mask=reseed_mask,
        store=store,
        hh_maps=hh_maps,
        corr_map=corr_map,
        ages=ages,
        lr_num=1,
        lr_den=10,
        ei_quota=(0.6, 0.4),
        long_range_ratio=0.25,
        reseed_rate=0.2,
        epoch=77,
        promote_cfg=pcfg,
    )

    # Promotions occurred and demotions removed the negative-corr edge
    assert out["promoted"] >= 1
    # Edge (1,5) demoted
    assert not store.exists(1, 5, 0)
