from __future__ import annotations

import unittest

from symbolicplastic_snn.utils.prng import SeedSpace, Stream


class TestRngSnapshot(unittest.TestCase):
    def test_stream_state_roundtrip(self):
        ss = SeedSpace(0x12345678)
        s = ss.derive("module=test", "tag=snapshot")
        first = [s.u64() for _ in range(3)]
        st = s.get_state()
        cont1 = [s.u64() for _ in range(3)]

        # Restore a new stream from saved state
        s2 = Stream(1)
        s2.set_state(st)
        cont2 = [s2.u64() for _ in range(3)]
        self.assertEqual(cont1, cont2)


if __name__ == "__main__":
    unittest.main()

