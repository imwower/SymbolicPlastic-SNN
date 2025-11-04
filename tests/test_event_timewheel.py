from __future__ import annotations

import unittest

from symbolicplastic_snn.schedule.timewheel import EventWheel


class TestEventTimeWheel(unittest.TestCase):
    def test_ordering_stable(self):
        w = EventWheel(8)
        # Same tick inserts in arbitrary order but tick() must sort by seq
        w.schedule(0, "a")
        w.schedule(0, "b")
        w.schedule(0, "c")
        out = [e.payload for e in w.tick()]
        self.assertEqual(out, ["a", "b", "c"])  # seq increasing

    def test_wrap_and_delays(self):
        w = EventWheel(4)
        w.schedule(3, "x")  # lands at (0+3)%4=3
        out0 = [e.payload for e in w.tick()]  # 0
        self.assertEqual(out0, [])
        out1 = [e.payload for e in w.tick()]  # 1
        self.assertEqual(out1, [])
        out2 = [e.payload for e in w.tick()]  # 2
        self.assertEqual(out2, [])
        out3 = [e.payload for e in w.tick()]  # 3
        self.assertEqual(out3, ["x"])
        # Wrap around
        w.schedule(4, "y")  # from current=0, lands at slot 0 next cycle
        out4 = [e.payload for e in w.tick()]
        self.assertEqual(out4, ["y"])

    def test_negative_delay_raises(self):
        w = EventWheel(8)
        with self.assertRaises(ValueError):
            w.schedule(-1, None)


if __name__ == "__main__":
    unittest.main()
