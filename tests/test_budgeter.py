from __future__ import annotations

import unittest

from symbolicplastic_snn.realtime.budgeter import Budgeter, admit, reset


class TestBudgeter(unittest.TestCase):
    def test_admit_and_carry(self):
        b = Budgeter(per_step=10, carry=0)
        g, b = admit(b, 8)
        self.assertEqual(g, 8)
        self.assertEqual(b.carry, 2)  # 2 left over
        g, b = admit(b, 15)  # avail 12
        self.assertEqual(g, 12)
        self.assertEqual(b.carry, 0)

    def test_saturation(self):
        b = Budgeter(per_step=10, carry=50)
        g, b = admit(b, 0)  # avail saturates at 10 + min(carry,per_step)
        self.assertEqual(b.carry, 10)

    def test_reset(self):
        b = Budgeter(per_step=5, carry=3)
        b2 = reset(b)
        self.assertEqual(b2.carry, 0)
        self.assertEqual(b2.per_step, 5)


if __name__ == "__main__":
    unittest.main()

