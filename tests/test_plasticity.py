from __future__ import annotations

import unittest

from plasticity import PlasticityConfig, StructuralPlasticity
from topology import TopologyParams
import numpy as np
from symbolicplastic_snn.plasticity.stats import BitWindow, update_corr
from symbolicplastic_snn.plasticity.update import reweight_alias_smallstep


class StubSampler:
    def __init__(self):
        self.pool = {
            0: [(3, 1), (2, 1)],
            1: [(0, 1)],
            2: [(0, 1)],
            3: [(1, 1)],
        }

    def sample(self, post, count, sign, forbidden=None):  # pragma: no cover - simple stub
        forbidden = forbidden or set()
        chosen = []
        for pre, delay in self.pool.get(post, []):
            if pre in forbidden:
                continue
            chosen.append((pre, delay))
            if len(chosen) >= count:
                break
        return chosen


class PlasticityTests(unittest.TestCase):
    def test_prune_and_regrow(self):
        exc_in = [[(1, 1), (2, 1)], [], [], []]
        inh_in = [[(2, 1)], [], [], []]
        sampler = StubSampler()
        params = TopologyParams(k_in=3, ei_ratio=1.0, sigma=1.0)
        config = PlasticityConfig.from_topology_params(
            params, prune_quota=0.5, stdp_window=20
        )
        engine = StructuralPlasticity(exc_in, inh_in, sampler, config)
        engine.tracker.bump(post=0, pre=1, sign="exc", value=5.0)
        engine.tracker.bump(post=0, pre=2, sign="exc", value=-1.0)

        engine.run_cycle()

        self.assertEqual(len(engine.exc_in[0]), config.target_exc)
        self.assertEqual(len(engine.inh_in[0]), config.target_inh)
        pres = {pre for pre, _ in engine.exc_in[0]}
        self.assertIn(3, pres)
        self.assertNotIn(2, pres)


def test_bitwindow_shift_and_history():
    bw = BitWindow(width=8)
    # Use single-neuron mask for clarity
    bw.push(np.array([True], dtype=bool))   # 0000 0001
    bw.push(np.array([False], dtype=bool))  # 0000 0010
    bw.push(np.array([True], dtype=bool))   # 0000 0101
    assert int(bw.get_history(0)) == 0b00000101


def test_corr_sign_pre_before_post_positive():
    # Pre at t0, Post at t0+delta => positive correlation
    delta = 2
    pre = 0
    post = 0
    bw_pre = BitWindow(width=8)
    bw_post = BitWindow(width=8)
    # t0: pre fires
    bw_pre.push(np.array([True], dtype=bool))
    bw_post.push(np.array([False], dtype=bool))
    # t0+1: none
    bw_pre.push(np.array([False], dtype=bool))
    bw_post.push(np.array([False], dtype=bool))
    # t0+2: post fires
    bw_pre.push(np.array([False], dtype=bool))
    bw_post.push(np.array([True], dtype=bool))
    pre = int(bw_pre.get_history(0))
    post = int(bw_post.get_history(0))
    d = int(update_corr(np.uint32(pre), np.uint32(post), delta=2, alpha_pos=np.uint8(2), alpha_neg=np.uint8(1)))
    assert d > 0

    # Post precedes pre => negative correlation
    bw_pre2 = BitWindow(width=8)
    bw_post2 = BitWindow(width=8)
    # t0: post fires
    bw_pre2.push(np.array([False], dtype=bool))
    bw_post2.push(np.array([True], dtype=bool))
    # t0+1: none
    bw_pre2.push(np.array([False], dtype=bool))
    bw_post2.push(np.array([False], dtype=bool))
    # t0+2: pre fires
    bw_pre2.push(np.array([True], dtype=bool))
    bw_post2.push(np.array([False], dtype=bool))
    pre2 = int(bw_pre2.get_history(0))
    post2 = int(bw_post2.get_history(0))
    d2 = int(update_corr(np.uint32(pre2), np.uint32(post2), delta=2, alpha_pos=np.uint8(2), alpha_neg=np.uint8(1)))
    assert d2 < 0


def test_reweight_sum_and_quotas_respected():
    n = 8
    # Start uniform over n entries summing to 65535
    base = 65535 // n
    prob = np.full(n, base, dtype=np.uint16)
    rem = 65535 - base * n
    if rem > 0:
        prob[:rem] = (prob[:rem].astype(np.uint32) + 1).astype(np.uint16)

    # Correlation nudges: encourage first half (E), discourage second half (I)
    corr = np.array([5, 4, 3, 2, -1, -2, -3, -4], dtype=np.int32)

    out = reweight_alias_smallstep(
        prob_q016=prob,
        corr_int32=corr,
        lr_num=1,
        lr_den=10,
        keep_sum=True,
        ei_quota=(0.7, 0.3),
        long_range_ratio=0.25,
    )

    assert out.dtype == np.uint16
    total = int(out.astype(np.uint32).sum())
    assert total == 65535
    mid = n // 2
    sum_E = int(out[:mid].astype(np.uint32).sum())
    sum_I = int(out[mid:].astype(np.uint32).sum())
    # E/I quotas approximately satisfied (within 1 due to integer rounding)
    target_E = int(0.7 * total)
    assert abs(sum_E - target_E) <= 1
    # Long range = last quarter
    lr_start = n - max(1, n // 4)
    sum_LR = int(out[lr_start:].astype(np.uint32).sum())
    assert sum_LR <= int(0.25 * total) + 1


if __name__ == "__main__":
    unittest.main()
