from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.readout.readout import Readout, ReadoutConfig
from symbolicplastic_snn.utils.prng import SeedSpace


class TestReadoutTieRng(unittest.TestCase):
    def test_tie_break_is_deterministic_with_stream(self):
        rc = ReadoutConfig(window=1, early_exit=False)
        r1 = Readout(rc)
        r1.add_channel("A", np.array([0], dtype=np.int32))
        r1.add_channel("B", np.array([1], dtype=np.int32))
        ss = SeedSpace(42)
        tie = ss.derive("module=readout", "tie", "test")
        r1.set_tie_rng(tie)

        # Construct masks that cause ties between A and B every step
        masks = [
            np.array([1, 1], dtype=bool),
            np.array([1, 1], dtype=bool),
            np.array([1, 1], dtype=bool),
        ]
        labels_1 = []
        for m in masks:
            r1.step(m)
            out = r1.emit()
            labels_1.append(out["label"])  # label is name of channel

        # Repeat with a fresh Readout + same tie stream derivation -> identical sequence
        r2 = Readout(ReadoutConfig(window=1, early_exit=False))
        r2.add_channel("A", np.array([0], dtype=np.int32))
        r2.add_channel("B", np.array([1], dtype=np.int32))
        tie2 = SeedSpace(42).derive("module=readout", "tie", "test")
        r2.set_tie_rng(tie2)
        labels_2 = []
        for m in masks:
            r2.step(m)
            out = r2.emit()
            labels_2.append(out["label"])

        self.assertEqual(labels_1, labels_2)


if __name__ == "__main__":
    unittest.main()

