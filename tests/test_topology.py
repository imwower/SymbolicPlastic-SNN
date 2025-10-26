from __future__ import annotations

import random
import unittest

from topology import TopologyParams, build_small_world_topology


class TopologyTests(unittest.TestCase):
    def test_small_world_builder_respects_k(self):
        coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        layers = [0, 0, 1, 1]
        params = TopologyParams(k_in=4, sigma=1.0, alpha_layer=1.0, long_range_ratio=0.0)
        topo = build_small_world_topology(coords, layers, params, rng=random.Random(42))

        for neuron in range(len(coords)):
            exc_count = topo.exc_in.indptr[neuron + 1] - topo.exc_in.indptr[neuron]
            inh_count = topo.inh_in.indptr[neuron + 1] - topo.inh_in.indptr[neuron]
            self.assertEqual(exc_count + inh_count, params.k_in)
            for _, delay in topo.exc_in.iter_row(neuron):
                self.assertGreaterEqual(delay, 1)
            for _, delay in topo.inh_in.iter_row(neuron):
                self.assertGreaterEqual(delay, 1)


if __name__ == "__main__":
    unittest.main()
