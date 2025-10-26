from __future__ import annotations

import random
import unittest

from core.simulator import EventDrivenLIF, SynapseTopology


def build_topology():
    exc = [
        [],  # neuron 0 has no incoming excitatory edges
        [(0, 1)],  # neuron 1 receives from neuron 0 with delay 1
    ]
    inh = [
        [],
        [],
    ]
    return SynapseTopology.from_incoming_lists(exc, inh, num_neurons=2)


class SimulatorTests(unittest.TestCase):
    def test_single_spike_propagation(self):
        topology = build_topology()
        sim = EventDrivenLIF(topology, theta=1.0, tau_ref=2, lambda_=1.0)
        sim.state.potentials[0] = 1.0

        first = sim.step([0])
        self.assertEqual(first.spikes, [1, 0])
        self.assertEqual(first.next_active, [1])
        self.assertEqual(sim.state.refractory[0], 2)

        second = sim.step(first.next_active)
        self.assertEqual(second.spikes, [0, 1])
        self.assertEqual(second.next_active, [])
        self.assertEqual(sim.state.refractory[1], 2)

    def test_refractory_blocks_follow_up_spike(self):
        exc = [
            [(0, 1)],  # self-loop with delay 1
        ]
        inh = [
            [],
        ]
        topology = SynapseTopology.from_incoming_lists(exc, inh, num_neurons=1)
        sim = EventDrivenLIF(topology, theta=1.0, tau_ref=2, lambda_=1.0)
        sim.state.potentials[0] = 1.0

        first = sim.step([0])
        self.assertEqual(first.spikes, [1])
        self.assertEqual(first.next_active, [0])
        self.assertEqual(sim.state.refractory[0], 2)

        second = sim.step(first.next_active)
        self.assertEqual(second.spikes, [0])
        self.assertEqual(sim.state.refractory[0], 1)
        self.assertEqual(second.next_active, [])

    def test_seed_from_noise_probability_one(self):
        topology = build_topology()
        sim = EventDrivenLIF(topology, theta=1.0, tau_ref=1, lambda_=1.0)
        active = sim.seed_from_noise(rate_hz=1000, rng=random.Random(0))
        self.assertEqual(active, [0, 1])


if __name__ == "__main__":
    unittest.main()
