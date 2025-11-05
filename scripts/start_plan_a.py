from __future__ import annotations

import argparse
import json

import numpy as np

from symbolicplastic_snn.runner.loop import RunnerConfig, SnnRunner
from symbolicplastic_snn.schedule.timewheel import TimeWheel
from symbolicplastic_snn.core.prng import FloatRng


def build_plan_a_config() -> RunnerConfig:
    # 方案 A（稳妥大规模）
    rc = RunnerConfig(
        n_tiles=64,
        tile_size=32768,
        indices_per_event=8,
        slots=16,
        readout_window=100,
        refractory_steps=2,
        budget_per_step=160_000,
        core_ratio=(8 / 64.0),
        explore_ratio=(1 / 64.0),
        rate_max=0.002,
        promote_per_pre_cap=32,
    )
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Start training with Plan A parameters (deterministic).")
    ap.add_argument("--steps", type=int, default=2000, help="Number of steps to run")
    ap.add_argument("--seed", type=int, default=1234, help="Run seed for inputs and runner")
    ap.add_argument("--bytes-cap", type=int, default=2_147_483_648, help="TimeWheel bytes cap (e.g., 2 GiB)")
    ap.add_argument("--period", type=int, default=500, help="Metrics print period")
    args = ap.parse_args()

    rc = build_plan_a_config()
    runner = SnnRunner(rc, seed=args.seed)
    # Recreate TimeWheel with memory cap for fine-grained aggregation
    runner.wheel = TimeWheel(slots=rc.slots, bytes_cap=int(args.bytes_cap))

    rng = FloatRng(args.seed)
    for t in range(int(args.steps)):
        # Deterministic [0,1) float input per step
        x = rng.random(runner.N, dtype=np.float32)
        out = runner.step(x)
        runner.wheel.tick()

        if (t + 1) % int(args.period) == 0:
            m = runner.last_metrics or {}
            print(
                json.dumps(
                    {
                        "step": t + 1,
                        "N": int(runner.N),
                        "used_budget_ratio": float(m.get("used_budget_ratio", m.get("budget_used_ratio", 0.0))),
                        "deferred_events": int(m.get("deferred_events", m.get("deferred_count", 0))),
                        "emitted_core_events": int(m.get("emitted_core_events", 0)),
                        "emitted_explore_events": int(m.get("emitted_explore_events", 0)),
                    }
                )
            )

    print(json.dumps({"done": True, "steps": int(args.steps), "N": int(runner.N)}))


if __name__ == "__main__":
    main()

