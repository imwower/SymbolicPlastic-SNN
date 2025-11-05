from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.runner.loop import RunnerConfig, SnnRunner
from symbolicplastic_snn.core.prng import FloatRng


class TestCISmoke(unittest.TestCase):
    def test_minimal_run(self):
        rc = RunnerConfig(n_tiles=1, tile_size=8, indices_per_event=2, slots=4, readout_window=5)
        rc.pipeline_enabled = False
        r = SnnRunner(rc, seed=123)
        rng = FloatRng(123)
        for _ in range(5):
            x = rng.random(r.N, dtype=np.float32)
            r.step(x)
            r.wheel.tick()
        # Just ensure it runs and produces a label
        out = r.readout.emit()
        self.assertIn("label", out)


if __name__ == "__main__":
    unittest.main()

