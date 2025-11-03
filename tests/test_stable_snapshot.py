import os
import tempfile

import numpy as np

from symbolicplastic_snn.plasticity.stable_store import StableStore, StableEdge
from symbolicplastic_snn.io.stable_snapshot import save_stable, load_stable
from symbolicplastic_snn.io.checkpoint import save_checkpoint, load_checkpoint
from symbolicplastic_snn.runner.loop import SnnRunner, RunnerConfig


def _make_store():
    st = StableStore(per_pre_cap=8)
    st.add(StableEdge(pre_id=1, post_id=10, sign=np.int8(1), delay=np.uint8(0)))
    st.add(StableEdge(pre_id=1, post_id=11, sign=np.int8(-1), delay=np.uint8(1)))
    st.add(StableEdge(pre_id=2, post_id=5, sign=np.int8(1), delay=np.uint8(0)))
    return st


def test_save_load_stable_roundtrip():
    st = _make_store()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "stable.bin")
        save_stable(path, st)
        st2 = load_stable(path)
    # Compare contents
    def iter_edges(s):
        return sorted([(e.pre_id, e.post_id, int(e.sign), int(e.delay), int(e.state)) for e in s.iter_all()])
    assert iter_edges(st) == iter_edges(st2)


def test_checkpoint_restore_keeps_predictions():
    cfg = RunnerConfig(n_tiles=2, tile_size=8, indices_per_event=4, slots=6, readout_window=10, rate_max=1.0, theta=2, refractory_steps=1, budget_per_step=64)
    runner = SnnRunner(cfg, seed=42)
    # Add a stable edge to influence future steps
    runner.stable_store.add(StableEdge(pre_id=0, post_id=9, sign=np.int8(1), delay=np.uint8(0)))

    # One step with deterministic input
    x = np.zeros(runner.N, dtype=np.float32)
    x[: runner.N // 2] = 1.0
    runner.step(x)
    out1 = runner.readout.emit()

    # Save combined checkpoint
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ckpt.bin")
        alias = {"prob": runner.alias_corr_prob, "alias": np.arange(runner.n_tiles, dtype=np.int32)}
        save_checkpoint(p, runner.v, runner.ref, runner.seeds_core, alias, runner.stable_store, {"step_index": runner._step_index})
        v2, ref2, seeds2, alias2, stable2, rng_state = load_checkpoint(p)

    # Restore into a fresh runner
    runner2 = SnnRunner(cfg, seed=42)
    runner2.v = v2.astype(np.int16)
    runner2.ref = ref2.astype(np.uint8)
    runner2.seeds_core = seeds2.astype(np.uint64)
    runner2.alias_corr_prob = alias2["prob"].astype(np.uint16)
    runner2.stable_store = stable2

    runner2.step(x)
    out2 = runner2.readout.emit()
    assert out1["label"] == out2["label"]
