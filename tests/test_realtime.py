import numpy as np

from symbolicplastic_snn.conn.alias import build_alias, AliasForTile
from symbolicplastic_snn.runner.loop import SnnRunner, RunnerConfig


def _bias_alias_to_first_half(runner: SnnRunner) -> None:
    ntiles = runner.n_tiles
    w = np.zeros(ntiles, dtype=np.uint16)
    half = ntiles // 2
    if half == 0:
        half = 1
    base = 65535 // half
    w[:half] = base
    rem = 65535 - base * half
    if rem > 0:
        w[:rem] = (w[:rem].astype(np.uint32) + 1).astype(np.uint16)
    prob, alias = build_alias(w)
    runner.alias_tbl = AliasForTile(prob=prob, alias=alias)


def test_budget_enforced():
    cfg = RunnerConfig(n_tiles=4, tile_size=8, indices_per_event=8, slots=8, readout_window=20, rate_max=1.0, theta=1, refractory_steps=1, budget_per_step=4, early_exit=False)
    runner = SnnRunner(cfg, seed=0)
    # Drive all inputs to guarantee spikes
    x = np.ones(runner.N, dtype=np.float32)
    runner.step(x)
    # Defer should be positive due to very small budget
    assert runner.last_metrics["deferred_count"] > 0


def test_priority_order():
    cfg = RunnerConfig(n_tiles=4, tile_size=8, indices_per_event=4, slots=8, readout_window=20, rate_max=1.0, theta=1, refractory_steps=1, budget_per_step=4, early_exit=False)
    runner = SnnRunner(cfg, seed=1)
    # Bias alias so most events target first-half tiles (leader channel = 0 by default)
    _bias_alias_to_first_half(runner)

    # Single step with spikes to generate events
    x = np.ones(runner.N, dtype=np.float32)
    runner.step(x)

    pbp = runner.last_metrics["processed_by_priority"]
    # Budget small enough to only process a subset; ensure P0 handled first, P1/P2 likely zero
    assert pbp[0] >= 0
    # For stringent ordering, ensure no budget left for P1/P2
    assert pbp[1] == 0 and pbp[2] == 0


def test_early_exit_path():
    cfg = RunnerConfig(n_tiles=2, tile_size=8, indices_per_event=4, slots=6, readout_window=10, rate_max=1.0, theta=1, refractory_steps=1, budget_per_step=16, early_exit=True)
    runner = SnnRunner(cfg, seed=2)
    # Step with high input to latch earliest label and trigger early-exit
    x = np.ones(runner.N, dtype=np.float32)
    out = runner.step(x)
    assert out is not None
    assert out["label"] in ("C0", "C1")


def test_core_explore_quota_metrics():
    # Pure core
    cfg_core = RunnerConfig(n_tiles=4, tile_size=8, indices_per_event=4, slots=6, readout_window=10, rate_max=1.0, theta=1, refractory_steps=1, budget_per_step=64, early_exit=False)
    cfg_core.core_ratio = 1.0
    cfg_core.explore_ratio = 0.0
    r1 = SnnRunner(cfg_core, seed=3)
    x = np.ones(r1.N, dtype=np.float32)
    r1.step(x)
    m1 = r1.last_metrics
    assert m1["quota_core_tiles"] > 0 and m1["quota_explore_tiles"] == 0

    # Pure explore
    cfg_exp = RunnerConfig(n_tiles=4, tile_size=8, indices_per_event=4, slots=6, readout_window=10, rate_max=1.0, theta=1, refractory_steps=1, budget_per_step=64, early_exit=False)
    cfg_exp.core_ratio = 0.0
    cfg_exp.explore_ratio = 1.0
    r2 = SnnRunner(cfg_exp, seed=3)
    r2.step(x)
    m2 = r2.last_metrics
    assert m2["quota_core_tiles"] == 0 and m2["quota_explore_tiles"] > 0
