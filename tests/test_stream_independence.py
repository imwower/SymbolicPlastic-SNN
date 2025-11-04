from __future__ import annotations

import unittest

from symbolicplastic_snn.utils.prng import SeedSpace


def _take(seq, n):
    return [seq.u64() for _ in range(n)]


class TestStreamIndependence(unittest.TestCase):
    def test_streams_independent_and_order_invariant(self):
        ss = SeedSpace(123456789)

        # Derive in different orders
        a1 = ss.derive("module=noise", "name=a")
        b1 = ss.derive("module=noise", "name=b")
        c1 = ss.derive("module=noise", "name=c")

        ss2 = SeedSpace(123456789)
        b2 = ss2.derive("module=noise", "name=b")
        c2 = ss2.derive("module=noise", "name=c")
        a2 = ss2.derive("module=noise", "name=a")

        # Per-key sequences equal across orders
        self.assertEqual(_take(a1, 5), _take(a2, 5))
        self.assertEqual(_take(b1, 5), _take(b2, 5))
        self.assertEqual(_take(c1, 5), _take(c2, 5))

        # Different keys produce different sequences
        ss3 = SeedSpace(123456789)
        a3 = ss3.derive("module=noise", "name=a")
        b3 = ss3.derive("module=noise", "name=b")
        self.assertNotEqual(_take(a3, 5), _take(b3, 5))


if __name__ == "__main__":
    unittest.main()
