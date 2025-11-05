from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.readout.readout import Readout, ReadoutConfig
from symbolicplastic_snn.core.prng import SeedSpace


class TestReadoutTiebreakAndWTA(unittest.TestCase):
    def test_tiebreak_stable_with_seedspace(self):
        rc = ReadoutConfig(window=1, early_exit=False)
        r1 = Readout(rc)
        r1.add_channel("A", np.array([0], dtype=np.int32))
        r1.add_channel("B", np.array([1], dtype=np.int32))
        ss = SeedSpace(42)
        tie = ss.derive("module=readout", "tie", "test2")
        r1.set_tie_rng(tie)

        masks = [np.array([1, 1], dtype=bool) for _ in range(3)]
        labels_1 = []
        for m in masks:
            r1.step(m)
            out = r1.emit()
            labels_1.append(out["label"])  # label is name of channel

        r2 = Readout(rc)
        r2.add_channel("A", np.array([0], dtype=np.int32))
        r2.add_channel("B", np.array([1], dtype=np.int32))
        r2.set_tie_rng(SeedSpace(42).derive("module=readout", "tie", "test2"))
        labels_2 = []
        for m in masks:
            r2.step(m)
            out = r2.emit()
            labels_2.append(out["label"])  # label is name of channel

        self.assertEqual(labels_1, labels_2)

    def test_wta_inhibition(self):
        # Channel A will win and inhibit B under WTA
        rc = ReadoutConfig(window=4, early_exit=False, wta=True, wta_threshold=1, wta_inhibit=1)
        r = Readout(rc)
        a_ids = np.array([0, 1, 2], dtype=np.int32)
        b_ids = np.array([3, 4, 5], dtype=np.int32)
        r.add_channel("A", a_ids)
        r.add_channel("B", b_ids)

        # Step 1: both sides 1 spike → no inhibition effect on same winners
        m1 = np.array([1, 0, 0, 1, 0, 0], dtype=bool)
        r.step(m1)
        # Step 2: A has 2 spikes, inhibits B by 2
        m2 = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
        r.step(m2)
        out = r.emit()
        self.assertEqual(out["label"], "A")


if __name__ == "__main__":
    unittest.main()

