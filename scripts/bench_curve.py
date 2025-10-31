from __future__ import annotations

import argparse
import csv
from typing import Any, Dict, List

from bench_throughput import build_runner  # reuse helper


def run_curve(args: argparse.Namespace) -> List[Dict[str, Any]]:
    budgets = args.budgets
    out: List[Dict[str, Any]] = []
    for b in budgets:
        args.budget = int(b)
        runner = build_runner(args)
        # Minimal per-budget run
        import numpy as np
        rng = np.random.default_rng(args.seed)
        spikes_total = 0
        processed_total = 0
        deferred_total = 0
        from time import perf_counter

        t0 = perf_counter()
        for t in range(args.steps):
            x = rng.random(runner.N, dtype=np.float32)
            runner.step(x)
            runner.wheel.tick()
            mt = runner.last_metrics
            spikes_total += int(mt.get("spikes_count", 0))
            processed_by_pri = mt.get("processed_by_priority", {}) or {}
            processed_total += int(sum(processed_by_pri.values()))
            deferred_total += int(mt.get("deferred_count", 0))
        t1 = perf_counter()
        total_time = t1 - t0
        steps_per_sec = args.steps / total_time if total_time > 0 else 0.0
        deferred_ratio = (deferred_total / (processed_total + deferred_total)) if (processed_total + deferred_total) > 0 else 0.0
        out.append(
            {
                "budget": int(b),
                "steps": args.steps,
                "steps_per_sec": steps_per_sec,
                "processed_total": processed_total,
                "deferred_total": deferred_total,
                "deferred_ratio": deferred_ratio,
                "spikes_total": spikes_total,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep budget settings and report throughput/deferred curve.")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--budgets", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--n-tiles", dest="n_tiles", type=int, default=8)
    ap.add_argument("--tile-size", dest="tile_size", type=int, default=16)
    ap.add_argument("--indices-per-event", dest="indices_per_event", type=int, default=8)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--readout-window", dest="readout_window", type=int, default=50)
    ap.add_argument("--theta", type=int, default=3)
    ap.add_argument("--refractory-steps", dest="refractory_steps", type=int, default=2)
    ap.add_argument("--near-radius", dest="near_radius", type=int, default=1)
    ap.add_argument("--near-wrap", dest="near_wrap", action="store_true")
    ap.add_argument("--preaggregate", action="store_true")
    ap.add_argument("--preaggregate-min-events", dest="preaggregate_min_events", type=int, default=64)
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    args = ap.parse_args()

    res = run_curve(args)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
            w.writeheader()
            w.writerows(res)
    else:
        import json
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

