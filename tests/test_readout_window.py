from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.readout.readout import WindowReadoutConfig, init_state, update, decide


class TestReadoutWindow(unittest.TestCase):
    def test_window_threshold_hold(self):
        cfg = WindowReadoutConfig(window_steps=5, threshold=3, hold_steps=2)
        st = init_state(num_classes=3, cfg=cfg)
        # Deterministic class stream: 0,1,1,1,1 -> expect class 1 halt after hold
        seq = [0, 1, 1, 1, 1]
        halted_at = None
        pred_at = None
        for i, c in enumerate(seq):
            st = update(st, c)
            pred, halted = decide(st, cfg)
            if halted and halted_at is None:
                halted_at = i
                pred_at = pred
        self.assertIsNotNone(halted_at)
        self.assertEqual(pred_at, 1)

    def test_invalid_inputs(self):
        cfg = WindowReadoutConfig(window_steps=5, threshold=3)
        with self.assertRaises(ValueError):
            init_state(0, cfg)
        st = init_state(2, cfg)
        with self.assertRaises(ValueError):
            update(st, -1)
        with self.assertRaises(ValueError):
            update(st, 2)


if __name__ == "__main__":
    unittest.main()

