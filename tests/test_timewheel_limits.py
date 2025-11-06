from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.schedule.timewheel import TimeWheel, BlockEvent


def ev(post_tile: int, idx: list[int], k: list[int], delay: int = 0) -> BlockEvent:
    return BlockEvent(post_tile=int(post_tile), indices=np.array(idx, dtype=np.int32), k=np.array(k, dtype=np.int16), delay=int(delay))


class TestTimeWheelLimits(unittest.TestCase):
    def test_drop_new_policy(self):
        tw = TimeWheel(slots=2, max_bucket_size=1, drop_policy="drop_new")
        # Push two keys into slot 0
        tw.push(ev(0, [1], [2], delay=0))
        tw.push(ev(1, [2], [1], delay=0))  # should be dropped (new)
        out = tw.pop()
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0].post_tile, out[0].delay), (0, 0))

    def test_drop_oldest_policy(self):
        tw = TimeWheel(slots=2, max_bucket_size=1, drop_policy="drop_oldest")
        tw.push(ev(0, [1], [2], delay=0))
        tw.push(ev(1, [2], [1], delay=0))  # should evict (0,0)
        out = tw.pop()
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0].post_tile, out[0].delay), (1, 0))

    def test_raise_policy(self):
        tw = TimeWheel(slots=2, max_bucket_size=1, drop_policy="raise")
        tw.push(ev(0, [1], [1], delay=0))
        with self.assertRaises(RuntimeError):
            tw.push(ev(1, [2], [1], delay=0))

    def test_wraparound_stability(self):
        tw = TimeWheel(slots=2)
        # Schedule at current slot and next
        tw.push(ev(0, [1], [1], delay=0))
        tw.push(ev(1, [2], [1], delay=1))
        a = tw.pop()
        self.assertEqual([(e.post_tile, e.delay) for e in a], [(0, 0)])
        tw.tick()
        b = tw.pop()
        self.assertEqual([(e.post_tile, e.delay) for e in b], [(1, 1)])
        # Wrap again
        tw.tick()
        c = tw.pop()
        self.assertEqual(c, [])


if __name__ == "__main__":
    unittest.main()

