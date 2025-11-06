from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from symbolicplastic_snn.runner.loop import RunnerConfig, SnnRunner
from symbolicplastic_snn.core.prng import FloatRng


def main() -> None:
    rc = RunnerConfig(n_tiles=2, tile_size=16, indices_per_event=4, slots=4, readout_window=8)
    rc.pipeline_enabled = False
    rc.budget_per_step = 2048
    runner = SnnRunner(rc, seed=1234)
    rng = FloatRng(1234)
    out_dir = Path("examples/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in range(128):
        x = rng.random(runner.N, dtype=np.float32)
        runner.step(x)
        runner.wheel.tick()
        if (t + 1) % 16 == 0:
            m = runner.last_metrics or {}
            print(json.dumps({
                "step": t + 1,
                "spikes": int(m.get("spikes_count", 0)),
                "used_budget_ratio": float(m.get("used_budget_ratio", m.get("budget_used_ratio", 0.0))),
                "deferred_events": int(m.get("deferred_events", m.get("deferred_count", 0))),
            }))

    # Save a few arrays
    np.save(out_dir / "v.npy", runner.v)
    np.save(out_dir / "ref.npy", runner.ref)
    np.save(out_dir / "alias_corr_prob.npy", runner.alias_corr_prob)
    print(json.dumps({"done": True, "N": int(runner.N)}))


if __name__ == "__main__":
    main()

