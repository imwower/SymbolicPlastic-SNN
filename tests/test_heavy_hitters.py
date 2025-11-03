import numpy as np

from symbolicplastic_snn.plasticity.heavy_hitters import SpaceSavingK
from symbolicplastic_snn.plasticity.counters import update_edge_usage


def test_space_saving_retrieves_frequent():
    hh = SpaceSavingK(capacity=3)
    seq = [1] * 20 + [2] * 12 + [3] * 8 + [4, 5, 6, 7, 8]
    for x in seq:
        hh.update(x)
    top = hh.topk()
    ids = {t[0] for t in top}
    # Frequent items in top-k; rare likely evicted
    assert 1 in ids and 2 in ids
    # The third might be 3, but due to replacement it could vary; ensure size bound
    assert len(top) <= 3


def test_capacity_bound():
    k = 4
    hh = SpaceSavingK(capacity=k)
    for x in range(100):
        hh.update(x)
    assert len(hh.topk()) <= k


def test_update_edge_usage_batch():
    pre = np.array([10, 10, 10, 11, 11, 10, 12], dtype=np.int32)
    post = np.array([1, 2, 1, 3, 3, 1, 5], dtype=np.int32)
    maps = {}
    update_edge_usage(pre, post, maps, capacity=8)

    # For pre=10, post=1 occurs 3 times
    top10 = maps[10].topk()
    d10 = {pid: cnt for pid, cnt, err in top10}
    assert d10.get(1, 0) >= 3  # space-saving count is upper bound, >= true count

    # For pre=11, only 3's were observed
    top11 = maps[11].topk()
    assert set(pid for pid, cnt, err in top11) == {3}

