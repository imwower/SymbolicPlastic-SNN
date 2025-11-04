from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict, Optional

import numpy as np

from symbolicplastic_snn.core.prng import FloatRng
from symbolicplastic_snn.io.config import load_cfg, validate_cfg
from symbolicplastic_snn.runner.loop import SnnRunner, RunnerConfig


def cfg_from_dict(d: Dict[str, Any]) -> RunnerConfig:
    # Minimal mapper; reuse logic from scripts/run_local where possible
    rc = RunnerConfig()
    if not d:
        return rc
    try:
        if "readout" in d and isinstance(d["readout"], dict):
            ro = d["readout"]
            if "window_steps" in ro:
                rc.readout_window = int(ro["window_steps"])
        if "realtime" in d and isinstance(d["realtime"], dict):
            rt = d["realtime"]
            if "budget_per_step" in rt:
                rc.budget_per_step = int(rt["budget_per_step"])
            if "early_exit" in rt:
                rc.early_exit = bool(rt["early_exit"])
        if "plasticity" in d and isinstance(d["plasticity"], dict):
            pl = d["plasticity"]
            if "low_contrib_frac" in pl:
                rc.low_contrib_frac = float(pl["low_contrib_frac"])
            if "reseed_rate" in pl:
                rc.reseed_rate = float(pl["reseed_rate"])
        if "pipeline" in d and isinstance(d["pipeline"], dict):
            ph = d["pipeline"]
            if "enabled" in ph:
                rc.pipeline_enabled = bool(ph["enabled"])
            if "period" in ph:
                rc.pipeline_period = int(ph["period"])
            if "promote_min_age" in ph:
                rc.promote_min_age = int(ph["promote_min_age"])
            if "promote_min_corr" in ph:
                rc.promote_min_corr = int(ph["promote_min_corr"])
            if "promote_min_hits" in ph:
                rc.promote_min_hits = int(ph["promote_min_hits"])
            if "promote_per_pre_cap" in ph:
                rc.promote_per_pre_cap = int(ph["promote_per_pre_cap"])
            if "demote_age" in ph:
                rc.demote_age = int(ph["demote_age"])
            if "demote_corr" in ph:
                rc.demote_corr = int(ph["demote_corr"])
            if "flip_sign_pos" in ph:
                rc.flip_sign_pos = int(ph["flip_sign_pos"])
            if "flip_sign_neg" in ph:
                rc.flip_sign_neg = int(ph["flip_sign_neg"])
    except Exception:
        pass
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SNN and write periodic metrics to CSV directly.")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--period", type=int, default=500, help="Sampling period for writing metrics rows")
    ap.add_argument("--csv-out", type=str, default="metrics.csv", help="Output CSV path")
    args = ap.parse_args()

    # Load config if provided
    rc = RunnerConfig()
    if args.config:
        cfg = load_cfg(args.config)
        validate_cfg(cfg)
        rc = cfg_from_dict(cfg)

    runner = SnnRunner(rc, seed=args.seed)
    rng = FloatRng(args.seed)

    fields = [
        "step",
        "promoted_count",
        "demoted_count",
        "sign_flip_count",
        "stable_edges_total",
        "frozen_count",
        "used_budget_ratio",
        "deferred_events",
    ]
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in range(int(args.steps)):
            x = rng.random(runner.N, dtype=np.float32)
            runner.step(x)
            runner.wheel.tick()
            if (t + 1) % int(args.period) == 0:
                m = runner.last_metrics or {}
                row = {
                    "step": t + 1,
                    "promoted_count": int(m.get("promoted_count", 0)),
                    "demoted_count": int(m.get("demoted_count", 0)),
                    "sign_flip_count": int(m.get("sign_flip_count", 0)),
                    "stable_edges_total": int(m.get("stable_edges_total", 0)),
                    "frozen_count": int(m.get("frozen_count", 0)),
                    "used_budget_ratio": float(m.get("used_budget_ratio", m.get("budget_used_ratio", 0.0))),
                    "deferred_events": int(m.get("deferred_events", m.get("deferred_count", 0))),
                }
                writer.writerow(row)


if __name__ == "__main__":
    main()

