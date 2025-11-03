from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from symbolicplastic_snn.core.prng import FloatRng

from symbolicplastic_snn.io.config import load_yaml, validate_config
from symbolicplastic_snn.runner.loop import SnnRunner, RunnerConfig
from symbolicplastic_snn.runner.metrics import MetricsTracker
from symbolicplastic_snn.io.stable_snapshot import save_stable, load_stable


def cfg_from_dict(d: Dict[str, Any]) -> RunnerConfig:
    rc = RunnerConfig()
    # Map a few common fields from config
    try:
        readout = d.get("readout", {})
        if "window_steps" in readout:
            rc.readout_window = int(readout["window_steps"])
        if "early_exit" in readout:
            rc.early_exit = bool(readout["early_exit"])  # allow override here
        if "max_future_gain_ratio" in readout:
            rc.readout_max_future_gain_ratio = float(readout["max_future_gain_ratio"])
        if "wta" in readout:
            rc.readout_wta = bool(readout["wta"])
        if "wta_threshold" in readout:
            rc.readout_wta_threshold = int(readout["wta_threshold"])
        if "wta_inhibit" in readout:
            rc.readout_wta_inhibit = int(readout["wta_inhibit"])
        # Optional realtime budget
        if "realtime" in d and isinstance(d["realtime"], dict):
            rt = d["realtime"]
            if "budget_per_step" in rt:
                rc.budget_per_step = int(rt["budget_per_step"])
            if "early_exit" in rt:
                rc.early_exit = bool(rt["early_exit"])
        # Plasticity block
        if "fixed_point" in d and isinstance(d["fixed_point"], dict):
            fp = d["fixed_point"]
            if "refractory_steps" in fp:
                rc.refractory_steps = int(fp["refractory_steps"])
        if "plasticity" in d and isinstance(d["plasticity"], dict):
            pl = d["plasticity"]
            if "period" in pl:
                rc.plasticity_period = int(pl["period"])
            if "lr_num" in pl:
                rc.lr_num = int(pl["lr_num"])
            if "lr_den" in pl:
                rc.lr_den = int(pl["lr_den"])
            if "corr_decay_period" in pl:
                rc.corr_decay_period = int(pl["corr_decay_period"])
            if "corr_decay_shift" in pl:
                rc.corr_decay_shift = int(pl["corr_decay_shift"])
        # Connectivity quotas
        if "connectivity" in d and isinstance(d["connectivity"], dict):
            cn = d["connectivity"]
            if "core_ratio" in cn:
                rc.core_ratio = float(cn["core_ratio"])
            if "explore_ratio" in cn:
                rc.explore_ratio = float(cn["explore_ratio"])
            if "core_long_range_ratio" in cn:
                rc.core_long_range_ratio = float(cn["core_long_range_ratio"])
            if "explore_long_range_ratio" in cn:
                rc.explore_long_range_ratio = float(cn["explore_long_range_ratio"])
            if "near_radius" in cn:
                rc.near_radius = int(cn["near_radius"])
            if "near_wrap" in cn:
                rc.near_wrap = bool(cn["near_wrap"])
            if "ei_mapping_mode" in cn:
                rc.ei_mapping_mode = str(cn["ei_mapping_mode"])  # 'half'|'alternating'|'custom'
            if "ei_tiles" in cn and isinstance(cn["ei_tiles"], list):
                rc.ei_tiles = [int(x) for x in cn["ei_tiles"]]
            if "preaggregate" in cn:
                rc.preaggregate = bool(cn["preaggregate"])
            if "preaggregate_min_events" in cn:
                rc.preaggregate_min_events = int(cn["preaggregate_min_events"])
        # Stability rules
        if "stability_rules" in d and isinstance(d["stability_rules"], dict):
            st = d["stability_rules"]
            if "forbid_short_EE_loops" in st:
                rc.stability_forbid_short_EE_loops = bool(st["forbid_short_EE_loops"])
            if "min_ee_delay" in st:
                rc.stability_min_ee_delay = int(st["min_ee_delay"])
            if "drop_short_EE" in st:
                rc.stability_drop_short_EE = bool(st["drop_short_EE"])
    except Exception:
        pass
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a small SNN locally and print per-step metrics.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config (optional)")
    ap.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed for inputs and runner")
    ap.add_argument("--save-prefix", type=str, default=None, help="Optional prefix to save runner state (includes stable store)")
    ap.add_argument("--load-prefix", type=str, default=None, help="Optional prefix to load runner state before running")
    ap.add_argument("--save-stable", type=str, default=None, help="Optional path to save only StableStore snapshot")
    ap.add_argument("--load-stable", type=str, default=None, help="Optional path to load StableStore snapshot before running")
    args = ap.parse_args()

    # Load config if provided
    rc = RunnerConfig()
    cfg: Optional[Dict[str, Any]] = None
    if args.config:
        cfg = load_yaml(args.config)
        validate_config(cfg)
        rc = cfg_from_dict(cfg)

    runner = SnnRunner(rc, seed=args.seed)
    if args.load_prefix:
        try:
            runner.load_state(args.load_prefix)
        except Exception as e:
            print(f"Warning: failed to load state: {e}")
    if args.load_stable:
        try:
            runner.stable_store = load_stable(args.load_stable)
        except Exception as e:
            print(f"Warning: failed to load stable store: {e}")

    rng = FloatRng(args.seed)
    tracker = MetricsTracker()
    for t in range(args.steps):
        # Synthetic input in [0,1)
        x = rng.random(runner.N, dtype=np.float32)
        out = runner.step(x)
        runner.wheel.tick()
        if out is None:
            out = runner.readout.emit()

        metrics = runner.last_metrics
        tracker.update_from_runner(runner)
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

        # Periodic metrics report
        if MetricsTracker.should_report(t + 1, period=500):
            agg = tracker.to_dict()
            phase = ""
            if agg.get("frozen_count", 0) and agg.get("frozen_count", 0) >= 1:
                phase = " 固化阶段"
            print(json.dumps({"periodic_metrics": agg, "note": phase}))

    if args.save_prefix:
        runner.save_state(args.save_prefix)
    if args.save_stable:
        try:
            save_stable(args.save_stable, runner.stable_store)
        except Exception as e:
            print(f"Warning: failed to save stable store: {e}")


if __name__ == "__main__":
    main()
