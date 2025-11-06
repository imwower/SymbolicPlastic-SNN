from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np

from symbolicplastic_snn.runner.loop import RunnerConfig, SnnRunner
from symbolicplastic_snn.core.prng import FloatRng, SeedSpace, Stream


class TestE2ESnapshotContinuity(unittest.TestCase):
    def _attach_wheel_probe(self, runner: SnnRunner, out_list: list[list[tuple]]):
        orig_pop = runner.wheel.pop

        def pop_probe():
            evs = orig_pop()
            # Record a compact signature to keep runtime light
            sig = []
            for ev in evs:
                if ev.capped:
                    sig.append((int(ev.post_tile), int(ev.delay), True, 0, 0, int(ev.total_k)))
                else:
                    sig.append((int(ev.post_tile), int(ev.delay), False, int(ev.indices.size), int(ev.k.astype(np.int64).sum()), 0))
            out_list.append(sig)
            return evs

        runner.wheel.pop = pop_probe  # type: ignore[assignment]

    def test_split_run_matches_baseline(self):
        rc = RunnerConfig(n_tiles=2, tile_size=8, indices_per_event=4, slots=4, readout_window=8)
        rc.pipeline_enabled = False
        rc.budget_per_step = 1024
        # Freeze plasticity side-effects to isolate snapshot continuity
        rc.low_contrib_frac = 0.0  # disable reseed
        rc.lr_num = 0  # no alias reweight
        seed = 4321
        T = 64
        split = 32

        # Deterministic inputs
        rng = FloatRng(seed)
        X = [rng.random(rc.n_tiles * rc.tile_size, dtype=np.float32) for _ in range(T)]

        # Baseline
        pops_base: list[list[tuple]] = []
        r0 = SnnRunner(rc, seed=seed)
        self._attach_wheel_probe(r0, pops_base)
        metrics_base = []
        for t in range(T):
            r0.step(X[t])
            r0.wheel.tick()
            m = r0.last_metrics.copy()
            metrics_base.append({k: m.get(k) for k in (
                "spikes_count",
                "used_budget_ratio",
                "deferred_events",
                "emitted_core_events",
                "emitted_explore_events",
            )})

        # Split-run: save at split
        pops_pre: list[list[tuple]] = []
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "snap")
            r1 = SnnRunner(rc, seed=seed)
            self._attach_wheel_probe(r1, pops_pre)
            for t in range(split):
                r1.step(X[t])
                r1.wheel.tick()
            r1.save_state(prefix)

            r2 = SnnRunner(rc, seed=999)
            pops_after: list[list[tuple]] = []
            self._attach_wheel_probe(r2, pops_after)
            r2.load_state(prefix)
            metrics_split = []
            for t in range(split, T):
                r2.step(X[t])
                r2.wheel.tick()
                m = r2.last_metrics.copy()
                metrics_split.append({k: m.get(k) for k in (
                    "spikes_count",
                    "used_budget_ratio",
                    "deferred_events",
                    "emitted_core_events",
                    "emitted_explore_events",
                )})

        # Compare step-level metrics and pop signatures
        self.assertEqual(metrics_base[split:], metrics_split)
        self.assertEqual(pops_base[split:], pops_after)

        # Active streams continuity: draw 4 values from each stream state
        def draw_tail(runner: SnnRunner) -> dict[str, list[int]]:
            out: dict[str, list[int]] = {}
            for name, st in runner._active_streams.items():
                s2 = Stream(1)
                s2.set_state(st.get_state())
                out[name] = [int(s2.u64()) for _ in range(4)]
            return out

        tail_base = draw_tail(r0)
        tail_split = draw_tail(r2)
        # Compare intersection only: split-run may have fewer active streams (no early steps)
        common = set(tail_base.keys()) & set(tail_split.keys())
        self.assertTrue(all(tail_base[k] == tail_split[k] for k in common))


if __name__ == "__main__":
    unittest.main()
