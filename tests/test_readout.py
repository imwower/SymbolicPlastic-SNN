from __future__ import annotations

import unittest

from readout import PerceptronReadout, WinnerTakeAll


class ReadoutTests(unittest.TestCase):
    def test_winner_take_all(self):
        wta = WinnerTakeAll(group_map=[0, 0, 1, 1])
        wta.observe([1, 0, 1, 1])
        self.assertEqual(wta.winner(), 1)
        wta.reset()
        self.assertIsNone(wta.winner())

    def test_perceptron_learns(self):
        model = PerceptronReadout(n_features=2, n_classes=2, lr=1.0)
        samples = [[1, 0], [0, 1]] * 5
        labels = [0, 1] * 5
        model.train_epoch(samples, labels)
        self.assertEqual(model.predict([1, 0]), 0)
        self.assertEqual(model.predict([0, 1]), 1)


if __name__ == "__main__":
    unittest.main()
