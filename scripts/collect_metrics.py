from __future__ import annotations

import argparse
import json
import re
import sys
from typing import List


def parse_lines(lines: List[str]):
    pattern = re.compile(r"\{\"periodic_metrics\":\s*(\{.*\})")
    for ln in lines:
        m = pattern.search(ln)
        if not m:
            continue
        try:
            obj = json.loads(ln)
            pm = obj.get("periodic_metrics", {})
            yield pm
        except Exception:
            try:
                pm = json.loads(m.group(1))
                yield pm
            except Exception:
                continue


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract periodic_metrics JSON entries from run_local output and write CSV.")
    ap.add_argument("logfile", nargs="?", default="-", help="Path to log file (or - for stdin)")
    ap.add_argument("--ascii", action="store_true", help="Also print simple ASCII chart of stable_edges_total")
    args = ap.parse_args()

    if args.logfile == "-":
        data = sys.stdin.read().splitlines()
    else:
        with open(args.logfile, "r", encoding="utf-8") as f:
            data = f.read().splitlines()

    rows = list(parse_lines(data))
    # CSV header
    print("step,promoted,demoted,flipped,stable_total,frozen,used_budget_ratio,deferred")
    for i, pm in enumerate(rows, start=1):
        step = i * 500  # run_local prints every 500 steps by default
        print(
            f"{step},{pm.get('promoted_count',0)},{pm.get('demoted_count',0)},{pm.get('sign_flip_count',0)},{pm.get('stable_edges_total',0)},{pm.get('frozen_count',0)},{pm.get('used_budget_ratio',0.0)},{pm.get('deferred_events',0)}"
        )

    if args.ascii and rows:
        # Simple ASCII bar for stable_total
        print("\nASCII chart: stable_edges_total over time")
        max_total = max(int(pm.get("stable_edges_total", 0)) for pm in rows) or 1
        scale = 50 / max_total
        for i, pm in enumerate(rows, start=1):
            step = i * 500
            val = int(pm.get("stable_edges_total", 0))
            bar = "#" * max(1, int(val * scale))
            print(f"{step:6d} | {bar} ({val})")


if __name__ == "__main__":
    main()

