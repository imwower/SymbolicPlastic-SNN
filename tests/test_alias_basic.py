from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.conn.alias import build_alias_table, sample_alias_stream, AliasSampler
from symbolicplastic_snn.utils.prng import SeedSpace


class TestAliasBasic(unittest.TestCase):
    def test_build_alias_validations(self):
        with self.assertRaises(ValueError):
            build_alias_table(np.array([]))
        with self.assertRaises(ValueError):
            build_alias_table(np.array([-0.1, 0.2]))
        with self.assertRaises(ValueError):
            build_alias_table(np.array([0.0, 0.0]))
        prob, alias = build_alias_table(np.array([1.0]))
        self.assertEqual(prob.shape[0], 1)
        self.assertEqual(alias[0], 0)

    def test_sample_alias_deterministic_first64(self):
        weights = np.array([0.1, 0.2, 0.7], dtype=float)
        prob, alias = build_alias_table(weights)
        ss = SeedSpace(0xBEEF)
        st = ss.derive("module=alias", "table=demo")
        out = sample_alias_stream(prob, alias, st, 64)
        # Golden generated once and fixed
        want = np.array([
            1,2,2,1,1,2,2,1,2,2,2,2,1,2,2,2,
            2,2,2,2,2,2,2,2,2,1,2,1,2,2,2,0,
            0,0,2,2,1,2,0,2,2,1,2,2,1,2,1,1,
            0,2,2,1,2,2,2,2,1,0,1,2,2,0,2,1,
        ], dtype=np.int32)
        self.assertTrue(np.array_equal(out, want))

    def test_sample_alias_empirical_freq(self):
        weights = np.array([0.2, 0.3, 0.5], dtype=float)
        prob, alias = build_alias_table(weights)
        ss = SeedSpace(12345)
        st = ss.derive("module=alias", "table=freq")
        n = 10000
        out = sample_alias_stream(prob, alias, st, n)
        hist = np.bincount(out, minlength=3) / float(n)
        self.assertLess(abs(hist[0] - 0.2), 0.015)
        self.assertLess(abs(hist[1] - 0.3), 0.015)
        self.assertLess(abs(hist[2] - 0.5), 0.015)


if __name__ == "__main__":
    unittest.main()
