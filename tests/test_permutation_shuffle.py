from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.utils.prng import SeedSpace


class TestPermutationShuffle(unittest.TestCase):
    def test_permutation_and_shuffle_determinism(self):
        ss1 = SeedSpace(777)
        ss2 = SeedSpace(777)

        s1 = ss1.derive("module=test", "perm=example")
        s2 = ss2.derive("module=test", "perm=example")

        p1 = s1.permutation(20)
        p2 = s2.permutation(20)
        self.assertTrue(np.array_equal(p1, p2))

        a1 = np.arange(30, dtype=np.int64)
        a2 = np.arange(30, dtype=np.int64)
        s3 = ss1.derive("module=test", "shuffle=list")
        s4 = ss2.derive("module=test", "shuffle=list")
        s3.shuffle(a1)
        s4.shuffle(a2)
        self.assertTrue(np.array_equal(a1, a2))


if __name__ == "__main__":
    unittest.main()
