from __future__ import annotations

import argparse
import csv
import json
from typing import Optional

import numpy as np

from symbolicplastic_snn.runner.loop import RunnerConfig, SnnRunner
from symbolicplastic_snn.schedule.timewheel import TimeWheel
from symbolicplastic_snn.core.prng import FloatRng


def build_plan_a_config(
    n_tiles: int | None = None,
    tile_size: int | None = None,
    indices_per_event: int | None = None,
    slots: int | None = None,
    readout_window: int | None = None,
    refractory_steps: int | None = None,
    budget_per_step: int | None = None,
    core_ratio: float | None = None,
    explore_ratio: float | None = None,
    rate_max: float | None = None,
    promote_per_pre_cap: int | None = None,
) -> RunnerConfig:
    # 方案 A（稳妥大规模）
    rc = RunnerConfig(
        n_tiles=int(n_tiles if n_tiles is not None else 64),
        tile_size=int(tile_size if tile_size is not None else 32768),
        indices_per_event=int(indices_per_event if indices_per_event is not None else 8),
        slots=int(slots if slots is not None else 16),
        readout_window=int(readout_window if readout_window is not None else 100),
        refractory_steps=int(refractory_steps if refractory_steps is not None else 2),
        budget_per_step=int(budget_per_step if budget_per_step is not None else 160_000),
        core_ratio=float(core_ratio if core_ratio is not None else (8 / 64.0)),
        explore_ratio=float(explore_ratio if explore_ratio is not None else (1 / 64.0)),
        rate_max=float(rate_max if rate_max is not None else 0.002),
        promote_per_pre_cap=int(promote_per_pre_cap if promote_per_pre_cap is not None else 32),
    )
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Start training with Plan A parameters (deterministic).")
    ap.add_argument("--steps", type=int, default=2000, help="Number of steps to run")
    ap.add_argument("--seed", type=int, default=1234, help="Run seed for inputs and runner")
    ap.add_argument("--bytes-cap", type=int, default=2_147_483_648, help="TimeWheel bytes cap (e.g., 2 GiB)")
    ap.add_argument("--period", type=int, default=500, help="Metrics print period")
    ap.add_argument("--csv-out", type=str, default=None, help="Optional path to write periodic metrics as CSV")
    # Optional overrides for development/smoke runs
    ap.add_argument("--n-tiles", type=int, default=None, help="Override number of tiles (default 64)")
    ap.add_argument("--tile-size", type=int, default=None, help="Override tile size (default 32768)")
    ap.add_argument("--indices-per-event", type=int, default=None, help="Override indices per event (default 8)")
    ap.add_argument("--budget-per-step", type=int, default=None, help="Override per-step budget (default 160000)")
    ap.add_argument("--rate-max", type=float, default=None, help="Override input rate_max (default 0.002)")
    args = ap.parse_args()

    rc = build_plan_a_config(
        n_tiles=args.n_tiles,
        tile_size=args.tile_size,
        indices_per_event=args.indices_per_event,
        budget_per_step=args.budget_per_step,
        rate_max=args.rate_max,
    )
    runner = SnnRunner(rc, seed=args.seed)
    # Recreate TimeWheel with memory cap for fine-grained aggregation
    runner.wheel = TimeWheel(slots=rc.slots, bytes_cap=int(args.bytes_cap))

    rng = FloatRng(args.seed)

    # Optional CSV writer for periodic metrics
    csv_writer: Optional[csv.DictWriter] = None
    csv_file = None
    if args.csv_out:
        csv_file = open(args.csv_out, "w", newline="", encoding="utf-8")
        fields = [
            "step",
            "N",
            "used_budget_ratio",
            "deferred_events",
            "emitted_core_events",
            "emitted_explore_events",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
        csv_writer.writeheader()

    for t in range(int(args.steps)):
        # Deterministic [0,1) float input per step
        x = rng.random(runner.N, dtype=np.float32)
        out = runner.step(x)
        runner.wheel.tick()

        if (t + 1) % int(args.period) == 0:
            m = runner.last_metrics or {}
            row = {
                "step": t + 1,
                "N": int(runner.N),
                "used_budget_ratio": float(m.get("used_budget_ratio", m.get("budget_used_ratio", 0.0))),
                "deferred_events": int(m.get("deferred_events", m.get("deferred_count", 0))),
                "emitted_core_events": int(m.get("emitted_core_events", 0)),
                "emitted_explore_events": int(m.get("emitted_explore_events", 0)),
            }
            # Stream JSONL for live monitoring
            print(json.dumps(row), flush=True)
            # Also write CSV if enabled
            if csv_writer is not None:
                csv_writer.writerow(row)
                csv_file.flush()

    print(json.dumps({"done": True, "steps": int(args.steps), "N": int(runner.N)}), flush=True)

    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
