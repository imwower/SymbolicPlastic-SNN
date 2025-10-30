from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from symbolicplastic_snn.io.config import load_yaml, validate_config
from symbolicplastic_snn.runner.loop import SnnRunner, RunnerConfig


def cfg_from_dict(d: Dict[str, Any]) -> RunnerConfig:
    rc = RunnerConfig()
    # Map a few common fields from config
    try:
        readout = d.get("readout", {})
        if "window_steps" in readout:
            rc.readout_window = int(readout["window_steps"])
    except Exception:
        pass
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a small SNN locally and print per-step metrics.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config (optional)")
    ap.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed for inputs and runner")
    args = ap.parse_args()

    # Load config if provided
    rc = RunnerConfig()
    cfg: Optional[Dict[str, Any]] = None
    if args.config:
        cfg = load_yaml(args.config)
        validate_config(cfg)
        rc = cfg_from_dict(cfg)

    runner = SnnRunner(rc, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    for t in range(args.steps):
        # Synthetic input in [0,1)
        x = rng.random(runner.N, dtype=np.float32)
        out = runner.step(x)
        runner.wheel.tick()
        if out is None:
            out = runner.readout.emit()

        metrics = runner.last_metrics
        print(
            json.dumps(
                {
                    "step": t,
                    "spikes": int(metrics.get("spikes_count", 0)),
                    "used_budget": int(metrics.get("budget_used", 0)),
                    "deferred": int(metrics.get("deferred_count", 0)),
                    "label": out.get("label", ""),
                    "latency": int(out.get("latency", -1)),
                }
            )
        )


if __name__ == "__main__":
    main()

