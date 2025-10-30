import numpy as np

from symbolicplastic_snn.runner.loop import SnnRunner, RunnerConfig


def test_smoke_small_network():
    cfg = RunnerConfig(n_tiles=4, tile_size=8, indices_per_event=4, slots=8, readout_window=20, rate_max=0.3, theta=3, refractory_steps=2, budget_per_step=256)
    runner = SnnRunner(cfg, seed=123)

    T = 100
    # Synthetic input: random in [0,1)
    X_T = [np.random.default_rng(0).random(runner.N).astype(np.float32) for _ in range(T)]
    out = runner.run(X_T)
    # Runner should produce output dict with label/scores
    assert isinstance(out, dict)
    assert "label" in out and "scores" in out


def test_readout_nonempty():
    cfg = RunnerConfig(n_tiles=2, tile_size=8, indices_per_event=4, slots=6, readout_window=15, rate_max=0.4, theta=2, refractory_steps=1, budget_per_step=128)
    runner = SnnRunner(cfg, seed=321)

    T = 60
    # Drive the first half more strongly to bias readout
    X_T = []
    for t in range(T):
        x = np.zeros(runner.N, dtype=np.float32)
        x[: runner.N // 2] = 1.0  # always fire at high rate
        x[runner.N // 2 :] = 0.1
        X_T.append(x)

    out = runner.run(X_T)
    assert out["label"] in ("C0", "C1")
    # Expect C0 often due to stronger drive; not strictly asserted

