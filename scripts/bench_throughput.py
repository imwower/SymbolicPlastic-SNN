from __future__ import annotations

import argparse
import json
import statistics as stats
from time import perf_counter
from typing import Any, Dict

import numpy as np

from symbolicplastic_snn.runner.loop import RunnerConfig, SnnRunner


def build_runner(args: argparse.Namespace) -> SnnRunner:
    rc = RunnerConfig(
        n_tiles=args.n_tiles,
        tile_size=args.tile_size,
        indices_per_event=args.indices_per_event,
        slots=args.slots,
        readout_window=args.readout_window,
        theta=args.theta,
        refractory_steps=args.refractory_steps,
        budget_per_step=args.budget,
        early_exit=args.early_exit,
    )
    rc.near_radius = args.near_radius
    rc.near_wrap = args.near_wrap
    rc.preaggregate = args.preaggregate
    rc.preaggregate_min_events = args.preaggregate_min_events
    return SnnRunner(rc, seed=args.seed)


def run_once(args: argparse.Namespace) -> Dict[str, Any]:
    runner = build_runner(args)
    rng = np.random.default_rng(args.seed)
    step_times = []
    spikes_total = 0
    processed_total = 0
    deferred_total = 0
    generated_total = 0

    t0 = perf_counter()
    for t in range(args.steps):
        x = rng.random(runner.N, dtype=np.float32)
        runner.step(x)
        runner.wheel.tick()
        mt = runner.last_metrics
        step_times.append(runner.step_durations[-1])
        spikes_total += int(mt.get("spikes_count", 0))
        processed_by_pri = mt.get("processed_by_priority", {}) or {}
        processed_total += int(sum(processed_by_pri.values()))
        deferred_total += int(mt.get("deferred_count", 0))
        generated_total += int(mt.get("emitted_core_events", 0)) + int(mt.get("emitted_explore_events", 0))
    t1 = perf_counter()

    total_time = t1 - t0
    steps_per_sec = args.steps / total_time if total_time > 0 else 0.0
    p50 = stats.median(step_times) if step_times else 0.0
    p90 = float(np.percentile(step_times, 90)) if step_times else 0.0
    p99 = float(np.percentile(step_times, 99)) if step_times else 0.0

    return {
        "steps": args.steps,
        "total_time_sec": total_time,
        "steps_per_sec": steps_per_sec,
        "p50_step_sec": p50,
        "p90_step_sec": p90,
        "p99_step_sec": p99,
        "spikes_total": spikes_total,
        "generated_events_total": generated_total,
        "processed_events_total": processed_total,
        "deferred_events_total": deferred_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark throughput and basic realtime metrics.")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-tiles", dest="n_tiles", type=int, default=8)
    ap.add_argument("--tile-size", dest="tile_size", type=int, default=16)
    ap.add_argument("--indices-per-event", dest="indices_per_event", type=int, default=8)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--readout-window", dest="readout_window", type=int, default=50)
    ap.add_argument("--theta", type=int, default=3)
    ap.add_argument("--refractory-steps", dest="refractory_steps", type=int, default=2)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--early-exit", dest="early_exit", action="store_true")
    ap.add_argument("--near-radius", dest="near_radius", type=int, default=1)
    ap.add_argument("--near-wrap", dest="near_wrap", action="store_true")
    ap.add_argument("--preaggregate", action="store_true")
    ap.add_argument("--preaggregate-min-events", dest="preaggregate_min_events", type=int, default=64)
    args = ap.parse_args()

    res = run_once(args)
    print(json.dumps(res))


if __name__ == "__main__":
    main()

