from __future__ import annotations

import math
import unittest
import numpy as np

from symbolicplastic_snn.encode import PoissonRateEncoder, LatencyRankEncoder
from symbolicplastic_snn.core.prng import SeedSpace


class TestPoissonRateEncoder(unittest.TestCase):
    def test_deterministic_and_rate_match(self):
        N, T = 256, 200
        rate = np.full(N, 0.2, dtype=np.float32)
        enc = PoissonRateEncoder(rate=rate, T=T)
        ss = SeedSpace(2025)

        out1 = enc.encode(ss, trial=0)
        out2 = enc.encode(SeedSpace(2025), trial=0)  # same seed/keys -> identical
        self.assertTrue(np.array_equal(out1, out2))

        # Statistical mean within 3 sigma of expected
        p = 0.2
        total = N * T
        mean = out1.sum() / float(total)
        var = p * (1.0 - p) / float(total)
        sigma = math.sqrt(var)
        self.assertLess(abs(mean - p), 3.0 * sigma + 1e-9)


class TestLatencyRankEncoder(unittest.TestCase):
    def test_tie_break_and_one_spike_per_neuron(self):
        x = np.array([0.5, 0.5, 0.2, 0.9, 0.9, 0.9], dtype=np.float32)
        enc = LatencyRankEncoder(window=4)
        ss = SeedSpace(7)
        out1 = enc.encode(x, ss, trial=1)
        out2 = enc.encode(x, SeedSpace(7), trial=1)
        self.assertTrue(np.array_equal(out1, out2))

        # Each neuron spikes exactly once
        self.assertTrue(np.all(out1.sum(axis=0) == 1))
        # Stronger intensities should have earlier or equal times compared to weaker (modulo wrap for >window)
        times = out1.argmax(axis=0)  # time index of spike per neuron
        # Top trio (0.9) should not be later than 0.5 group (modulo wrap uncertainty only if >window)
        trio_t = times[3:6].min()
        pair_t = times[0:2].max()
        self.assertLessEqual(trio_t, pair_t)


if __name__ == "__main__":
    unittest.main()

