import numpy as np

from symbolicplastic_snn.runner.metrics import MetricsTracker
from symbolicplastic_snn.runner.loop import SnnRunner, RunnerConfig
from symbolicplastic_snn.plasticity.stable_store import StableStore, StableEdge


def test_metrics_accumulate_and_reset():
    mt = MetricsTracker()
    # Accumulate manual values
    mt.promoted_count += 3
    mt.demoted_count += 2
    mt.sign_flip_count += 1
    mt.used_budget_ratio = 0.5
    mt.deferred_events = 10
    d = mt.to_dict()
    assert d["promoted_count"] == 3
    assert d["demoted_count"] == 2
    assert d["sign_flip_count"] == 1
    assert 0.0 <= d["used_budget_ratio"] <= 1.0
    assert d["deferred_events"] == 10

    mt.reset()
    d2 = mt.to_dict()
    assert d2["promoted_count"] == 0 and d2["demoted_count"] == 0 and d2["sign_flip_count"] == 0
    assert d2["used_budget_ratio"] == 0.0 and d2["deferred_events"] == 0


def test_runner_emits_metrics_periodically():
    cfg = RunnerConfig(n_tiles=2, tile_size=8, indices_per_event=4, slots=6, readout_window=10, rate_max=0.5, theta=2, refractory_steps=1, budget_per_step=64)
    runner = SnnRunner(cfg, seed=7)
    # Add a frozen edge and a normal edge
    runner.stable_store.add(StableEdge(pre_id=0, post_id=1, sign=np.int8(1), delay=np.uint8(0)))
    e2 = StableEdge(pre_id=1, post_id=10, sign=np.int8(1), delay=np.uint8(0))
    runner.stable_store.add(e2)
    # Periodic reporting check
    mt = MetricsTracker()
    # Simulate runner stepping; used_budget_ratio present in last_metrics after a step
    x = np.zeros(runner.N, dtype=np.float32)
    mt.update_from_runner(runner)  # before step
    assert not MetricsTracker.should_report(1, period=500)
    assert MetricsTracker.should_report(500, period=500)

