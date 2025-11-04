from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.schedule.timewheel import TimeWheel, BlockEvent


class TestTimeWheelSnapshot(unittest.TestCase):
    def test_snapshot_restore_equivalence(self):
        tw = TimeWheel(slots=4)
        # Push some events across different delays and tiles
        tw.push(BlockEvent(post_tile=0, indices=np.array([1, 3], dtype=np.int32), k=np.array([2, 1], dtype=np.int16), delay=0))
        tw.push(BlockEvent(post_tile=1, indices=np.array([0], dtype=np.int32), k=np.array([5], dtype=np.int16), delay=2))
        tw.push(BlockEvent(post_tile=0, indices=np.array([2], dtype=np.int32), k=np.array([3], dtype=np.int16), delay=3))
        tw.tick()
        tw.push(BlockEvent(post_tile=1, indices=np.array([2, 4], dtype=np.int32), k=np.array([1, 1], dtype=np.int16), delay=1))

        snap = tw.snapshot()

        tw2 = TimeWheel(slots=4)
        tw2.restore(snap)
        # Snapshots should match
        self.assertEqual(tw.snapshot(), tw2.snapshot())

        # Pop in lockstep for several ticks; events should match exactly
        for _ in range(4):
            a = tw.pop()
            b = tw2.pop()
            # Compare BlockEvent content
            self.assertEqual(len(a), len(b))
            for ev1, ev2 in zip(a, b):
                self.assertEqual(ev1.post_tile, ev2.post_tile)
                self.assertEqual(ev1.delay, ev2.delay)
                self.assertEqual(bool(ev1.capped), bool(ev2.capped))
                self.assertTrue(np.array_equal(ev1.indices, ev2.indices))
                self.assertTrue(np.array_equal(ev1.k, ev2.k))
            tw.tick(); tw2.tick()


if __name__ == "__main__":
    unittest.main()

