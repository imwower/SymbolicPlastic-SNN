from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.core.fixedpoint import q_add_sat, q_sub_sat, q_mul_q


class TestFixedPointOps(unittest.TestCase):
    def test_add_sat(self):
        self.assertEqual(int(q_add_sat(np.int16(30000), np.int16(10000))), 32767)
        self.assertEqual(int(q_add_sat(np.int16(-30000), np.int16(-10000))), -32768)

    def test_sub_sat(self):
        self.assertEqual(int(q_sub_sat(np.int16(-32768), np.int16(1))), -32768)
        self.assertEqual(int(q_sub_sat(np.int16(32767), np.int16(-2))), 32767)

    def test_mul_q_q411(self):
        # 1.0 (Q4.11=2048) * 0.5 (Q4.11=1024) => 0.5 (1024)
        self.assertEqual(int(q_mul_q(np.int16(2048), np.int16(1024), 11)), 1024)


if __name__ == "__main__":
    unittest.main()

