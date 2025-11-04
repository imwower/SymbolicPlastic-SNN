from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.encode.input_encoders import constant_q16, piecewise_linear, poisson_spikes, ratio_norm
from symbolicplastic_snn.utils.prng import SeedSpace


class TestEncoders(unittest.TestCase):
    def test_constant_and_ratio(self):
        self.assertEqual(int(constant_q16(0.5, 1.0)), 32768)
        self.assertEqual(int(ratio_norm(0.0, 0.0, 1.0)), 0)
        self.assertEqual(int(ratio_norm(1.0, 0.0, 1.0)), 65535)
        self.assertEqual(int(ratio_norm(0.5, 0.0, 1.0)), 32768)

    def test_piecewise(self):
        knots = np.array([0.0, 0.5, 1.0])
        values = np.array([0.0, 0.5, 1.0])
        self.assertEqual(int(piecewise_linear(0.25, knots, values)), 16384)
        self.assertEqual(int(piecewise_linear(0.75, knots, values)), 49152)

    def test_poisson_spikes_golden(self):
        ss = SeedSpace(0xFEED)
        st = ss.derive("module=encode", "poisson=test")
        seq = [poisson_spikes(50.0, 1.0, st) for _ in range(128)]
        # Golden once generated and fixed
        want = [
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
            False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
            False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False,
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False,
            False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False,
        ]
        self.assertEqual(seq, want)


if __name__ == "__main__":
    unittest.main()
