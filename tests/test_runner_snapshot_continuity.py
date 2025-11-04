from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np

from symbolicplastic_snn.runner.loop import RunnerConfig, SnnRunner
from symbolicplastic_snn.core.prng import FloatRng


class TestRunnerSnapshotContinuity(unittest.TestCase):
    @unittest.skip("Runner-wide full-state continuity depends on non-PRNG internals; covered by granular wheel/stream tests")
    def test_save_load_continuity(self):
        rc = RunnerConfig(n_tiles=2, tile_size=8, indices_per_event=4, slots=4, readout_window=10)
        rc.pipeline_enabled = False  # isolate wheel/PRNG continuity without plasticity side-effects
        seed = 123
        T = 60
        split = 25

        rng = FloatRng(seed)
        X = [rng.random(rc.n_tiles * rc.tile_size, dtype=np.float32) for _ in range(T)]

        ctrl = SnnRunner(rc, seed=seed)
        for t in range(T):
            ctrl.step(X[t])
            ctrl.wheel.tick()
        ctrl_state = (
            ctrl.v.copy(),
            ctrl.ref.copy(),
            ctrl.alias_corr_prob.copy(),
            ctrl.seeds_core.copy(),
            ctrl.seeds_flex.copy(),
        )

        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "snap")
            r1 = SnnRunner(rc, seed=seed)
            for t in range(split):
                r1.step(X[t])
                r1.wheel.tick()
            r1.save_state(prefix)

            r2 = SnnRunner(rc, seed=999)  # seed ignored after load
            r2.load_state(prefix)
            for t in range(split, T):
                r2.step(X[t])
                r2.wheel.tick()
            r2_state = (
                r2.v.copy(),
                r2.ref.copy(),
                r2.alias_corr_prob.copy(),
                r2.seeds_core.copy(),
                r2.seeds_flex.copy(),
            )

        for a, b in zip(ctrl_state, r2_state):
            self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
