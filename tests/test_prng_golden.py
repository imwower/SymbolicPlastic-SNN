from __future__ import annotations

import math
import unittest

from symbolicplastic_snn.utils.prng import splitmix64_next, SeedSpace


class TestPrngGolden(unittest.TestCase):
    def test_splitmix64_golden_sequence(self):
        seed = 0x0123456789ABCDEF
        s = seed
        outs = []
        for _ in range(4):
            out, s = splitmix64_next(s)
            outs.append(out)
        expected = [
            1547611027431991965,
            15380727978956804243,
            3427440727199435966,
            11733030637320693740,
        ]
        self.assertEqual(outs, expected)

    def test_seedspace_stream_golden(self):
        run_seed = 0x0123456789ABCDEF
        ss = SeedSpace(run_seed)
        st = ss.derive("module=noise", "layer=3", "conn=pre42->post7")
        xs = [st.u64() for _ in range(5)]
        expected_u64 = [
            9928154422538561351,
            6317639509273722700,
            4208093950076753653,
            12633194102282835104,
            10516725126782500425,
        ]
        self.assertEqual(xs, expected_u64)

        st2 = ss.derive("module=noise", "layer=3", "conn=pre42->post7")
        floats = [st2.uniform() for _ in range(3)]
        expected_f = [
            0.5382063296843937,
            0.3424799240467412,
            0.2281212301348161,
        ]
        for a, b in zip(floats, expected_f):
            self.assertTrue(math.isclose(a, b, rel_tol=1e-15, abs_tol=0.0))


if __name__ == "__main__":
    unittest.main()
