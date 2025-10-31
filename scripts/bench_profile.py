from __future__ import annotations

import argparse
import tracemalloc
from time import perf_counter
from typing import List

import numpy as np

from symbolicplastic_snn.runner.loop import RunnerConfig, SnnRunner


def main() -> None:
    ap = argparse.ArgumentParser(description="Memory profiling with tracemalloc (top alloc sites).")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-tiles", dest="n_tiles", type=int, default=8)
    ap.add_argument("--tile-size", dest="tile_size", type=int, default=16)
    ap.add_argument("--indices-per-event", dest="indices_per_event", type=int, default=8)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rc = RunnerConfig(
        n_tiles=args.n_tiles,
        tile_size=args.tile_size,
        indices_per_event=args.indices_per_event,
        slots=args.slots,
        budget_per_step=args.budget,
    )
    runner = SnnRunner(rc, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    tracemalloc.start()
    t0 = perf_counter()
    for t in range(args.steps):
        x = rng.random(runner.N, dtype=np.float32)
        runner.step(x)
        runner.wheel.tick()
    t1 = perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()

    print(f"elapsed_sec={t1 - t0:.6f} current_MB={current/1e6:.2f} peak_MB={peak/1e6:.2f}")
    stats = snap.statistics("lineno")[: args.top]
    for s in stats:
        size_mb = s.size / 1e6
        print(f"{size_mb:8.2f} MB {s.traceback.format()[-1] if s.traceback else s}")


if __name__ == "__main__":
    main()

