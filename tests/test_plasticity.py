from __future__ import annotations

import unittest

from plasticity import PlasticityConfig, StructuralPlasticity
from topology import TopologyParams


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


if __name__ == "__main__":
    unittest.main()
