from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np

from symbolicplastic_snn.runner.loop import RunnerConfig, SnnRunner
from symbolicplastic_snn.encode import PoissonRateEncoder
from symbolicplastic_snn.core.prng import SeedSpace
from symbolicplastic_snn.io.stable_snapshot import _store_to_arrays as _stable_to_arrays


class TestE2ESnapshotContinuity(unittest.TestCase):
    def test_split_run_matches_baseline(self):
        # Minimal but non-trivial config
        rc = RunnerConfig(n_tiles=2, tile_size=8, indices_per_event=4, slots=4, readout_window=8)
        rc.pipeline_enabled = False  # isolate state continuity without plasticity
        rc.rate_max = 1.0            # make binary inputs pass-through in runner encoding
        seed = 123
        T = 33
        split = 32

        N = rc.n_tiles * rc.tile_size
        ss = SeedSpace(seed)
        # Use PoissonRateEncoder to produce deterministic binary inputs per step
        enc = PoissonRateEncoder(rate=np.full(N, 0.1, dtype=np.float32), T=T)
        X = enc.encode(ss, trial=0)  # shape (T, N) int8

        # Baseline up to the final step we will check
        ctrl = SnnRunner(rc, seed=seed)
        for t in range(T):
            ctrl.step(X[t].astype(np.float32))
            ctrl.wheel.tick()

        # Split run with save/load at split and continue 1 step
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "snap")
            r1 = SnnRunner(rc, seed=seed)
            for t in range(split):
                r1.step(X[t].astype(np.float32))
                r1.wheel.tick()
            r1.save_state(prefix)

            r2 = SnnRunner(rc, seed=999)  # seed ignored after load
            r2.load_state(prefix)
            # Advance one step and compare with baseline
            for t in range(split, T):
                r2.step(X[t].astype(np.float32))
                r2.wheel.tick()

        # Compare critical states (structural + deterministic subsystems)
        self.assertTrue(np.array_equal(ctrl.alias_corr_prob, r2.alias_corr_prob))
        self.assertTrue(np.array_equal(ctrl.seeds_core, r2.seeds_core))

        # TimeWheel pointer equality (full bucket equality may depend on set ordering)
        self.assertEqual(ctrl.wheel.snapshot().get("ptr"), r2.wheel.snapshot().get("ptr"))

        # StableStore arrays
        ci, ce = _stable_to_arrays(ctrl.stable_store)
        ri, re = _stable_to_arrays(r2.stable_store)
        self.assertTrue(np.array_equal(ci, ri))
        self.assertTrue(np.array_equal(ce, re))

        # Readout buffers and metadata
        self.assertTrue(np.array_equal(getattr(ctrl.readout, "_hist"), getattr(r2.readout, "_hist")))
        self.assertTrue(np.array_equal(getattr(ctrl.readout, "_counts"), getattr(r2.readout, "_counts")))
        self.assertEqual(getattr(ctrl.readout, "_ptr"), getattr(r2.readout, "_ptr"))
        self.assertEqual(getattr(ctrl.readout, "_step_idx"), getattr(r2.readout, "_step_idx"))
        self.assertEqual(getattr(ctrl.readout, "_latched_idx"), getattr(r2.readout, "_latched_idx"))
        self.assertEqual(getattr(ctrl.readout, "_latency"), getattr(r2.readout, "_latency"))

        # Alias version and active streams (comparing states at the same step)
        self.assertEqual(getattr(ctrl, "_alias_version"), getattr(r2, "_alias_version"))
        self.assertEqual(set(ctrl._active_streams.keys()), set(r2._active_streams.keys()))


if __name__ == "__main__":
    unittest.main()
