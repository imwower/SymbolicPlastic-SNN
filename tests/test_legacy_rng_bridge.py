from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.utils.prng import SeedSpace
from core.simulator import EventDrivenLIF, SynapseTopology
from topology.generator import TopologyParams, build_small_world_topology


class LegacyBridgeTests(unittest.TestCase):
    def test_simulator_accepts_stream_rng(self):
        exc = [
            [],
            [(0, 1)],
        ]
        inh = [
            [],
            [],
        ]
        topo = SynapseTopology.from_incoming_lists(exc, inh, num_neurons=2)
        sim = EventDrivenLIF(topo, theta=1.0, tau_ref=1, lambda_=1.0)

        ss = SeedSpace(123)
        rng = ss.derive("module=sim", "test=seed_from_noise")
        # With high rate, should deterministically activate all
        act = sim.seed_from_noise(rate_hz=1000, rng=rng)
        self.assertEqual(act, [0, 1])

    def test_topology_accepts_stream_rng(self):
        coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        layers = [0, 0, 1, 1]
        params = TopologyParams(k_in=4, sigma=1.0, alpha_layer=1.0, long_range_ratio=0.0)
        ss = SeedSpace(999)
        rng = ss.derive("module=topology", "build=small_world")
        topo1 = build_small_world_topology(coords, layers, params, rng=rng)
        # Recreate with a fresh derived stream -> same result
        rng2 = ss.derive("module=topology", "build=small_world")
        topo2 = build_small_world_topology(coords, layers, params, rng=rng2)
        # Compare incoming counts
        for neuron in range(len(coords)):
            e1 = topo1.exc_in.indptr[neuron + 1] - topo1.exc_in.indptr[neuron]
            i1 = topo1.inh_in.indptr[neuron + 1] - topo1.inh_in.indptr[neuron]
            e2 = topo2.exc_in.indptr[neuron + 1] - topo2.exc_in.indptr[neuron]
            i2 = topo2.inh_in.indptr[neuron + 1] - topo2.inh_in.indptr[neuron]
            self.assertEqual((e1, i1), (e2, i2))


if __name__ == "__main__":
    unittest.main()

