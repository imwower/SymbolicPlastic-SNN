from __future__ import annotations

import argparse
import csv
from typing import List, Tuple


def load_metrics_csv(path: str) -> Tuple[List[int], List[int], List[int], List[int]]:
    steps: List[int] = []
    stable: List[int] = []
    promoted: List[int] = []
    demoted: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row.get("step", 0)))
                stable.append(int(row.get("stable_total", 0)))
                promoted.append(int(row.get("promoted", 0)))
                demoted.append(int(row.get("demoted", 0)))
            except Exception:
                continue
    return steps, stable, promoted, demoted


def to_svg_line(points: List[Tuple[float, float]], color: str, width: float = 2.0) -> str:
    if not points:
        return ""
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{d}" />\n'


def scale_points(xs: List[int], ys: List[int], x0: float, y0: float, w: float, h: float) -> List[Tuple[float, float]]:
    if not xs or not ys or len(xs) != len(ys):
        return []
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = 0, max(ys) if ys else 1
    dx = (max_x - min_x) or 1
    dy = (max_y - min_y) or 1
    out: List[Tuple[float, float]] = []
    for x, y in zip(xs, ys):
        sx = x0 + (x - min_x) * w / dx
        sy = y0 + h - (y - min_y) * h / dy  # invert y for SVG
        out.append((sx, sy))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot metrics CSV (stable_total, promoted, demoted) to an SVG file.")
    ap.add_argument("csv", help="Path to metrics.csv (from scripts/collect_metrics.py)")
    ap.add_argument("--out", default="metrics.svg", help="Output SVG path (default: metrics.svg)")
    ap.add_argument("--color-stable", default="#2a6fdb")
    ap.add_argument("--color-promoted", default="#2bad4b")
    ap.add_argument("--color-demoted", default="#d92b2b")
    ap.add_argument("--split-panels", action="store_true", help="Plot stable_total and (promoted/demoted) in two stacked panels")
    ap.add_argument("--xticks", type=int, default=5, help="Number of x-axis ticks")
    ap.add_argument("--yticks", type=int, default=6, help="Number of y-axis ticks per panel")
    args = ap.parse_args()

    steps, stable, promoted, demoted = load_metrics_csv(args.csv)
    if not steps:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("<svg xmlns='http://www.w3.org/2000/svg' width='640' height='360'><text x='10' y='20'>No data</text></svg>")
        return

    # Layout
    W, H = 800, (480 if not args.split_panels else 720)
    PAD_L, PAD_T, PAD_R, PAD_B = 60, 20, 20, 60
    plot_w = W - PAD_L - PAD_R
    if args.split_panels:
        plot_h_single = (H - PAD_T - PAD_B - 40) / 2
        panels = [
            (PAD_L, PAD_T, plot_w, plot_h_single),
            (PAD_L, PAD_T + plot_h_single + 40, plot_w, plot_h_single),
        ]
    else:
        plot_h = H - PAD_T - PAD_B
        panels = [(PAD_L, PAD_T, plot_w, plot_h)]

    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}'>\n"]
    # Panel 1: stable_total
    x0, y0, ww, hh = panels[0]
    pts_stable = scale_points(steps, stable, x0, y0, ww, hh)
    # Axes and grid for panel 1
    svg.append(f"<line x1='{x0}' y1='{y0+hh}' x2='{x0+ww}' y2='{y0+hh}' stroke='black' stroke-width='1' />\n")
    svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y0+hh}' stroke='black' stroke-width='1' />\n")
    # Y ticks panel 1
    max_y1 = max(stable) if stable else 1
    for i in range(1, args.yticks):
        yv = i * max_y1 / args.yticks
        _, py = scale_points([steps[0]], [int(yv)], x0, y0, ww, hh)[0]
        svg.append(f"<line x1='{x0}' y1='{py:.1f}' x2='{x0+ww}' y2='{py:.1f}' stroke='#eee' stroke-width='1' />\n")
    # X ticks (shared)
    min_x, max_x = min(steps), max(steps)
    dx = (max_x - min_x) or 1
    for i in range(args.xticks + 1):
        xv = min_x + i * dx / args.xticks
        px, _ = scale_points([int(xv)], [0], x0, y0, ww, hh)[0]
        svg.append(f"<line x1='{px:.1f}' y1='{y0+hh}' x2='{px:.1f}' y2='{y0+hh+5}' stroke='black' stroke-width='1' />\n")
        svg.append(f"<text x='{px:.1f}' y='{y0+hh+20:.1f}' text-anchor='middle' font-size='10'>{int(xv)}</text>\n")
    # Lines panel 1
    svg.append(to_svg_line(pts_stable, args.color_stable, 2.5))
    svg.append(f"<text x='{x0+10}' y='{y0+15}' font-size='12'>stable_total</text>\n")

    if args.split_panels:
        # Panel 2: promoted/demoted
        x1, y1, ww2, hh2 = panels[1]
        pts_prom = scale_points(steps, promoted, x1, y1, ww2, hh2)
        pts_demo = scale_points(steps, demoted, x1, y1, ww2, hh2)
        svg.append(f"<line x1='{x1}' y1='{y1+hh2}' x2='{x1+ww2}' y2='{y1+hh2}' stroke='black' stroke-width='1' />\n")
        svg.append(f"<line x1='{x1}' y1='{y1}' x2='{x1}' y2='{y1+hh2}' stroke='black' stroke-width='1' />\n")
        max_y2 = max(promoted + demoted) if (promoted or demoted) else 1
        for i in range(1, args.yticks):
            yv = i * max_y2 / args.yticks
            _, py = scale_points([steps[0]], [int(yv)], x1, y1, ww2, hh2)[0]
            svg.append(f"<line x1='{x1}' y1='{py:.1f}' x2='{x1+ww2}' y2='{py:.1f}' stroke='#eee' stroke-width='1' />\n")
        svg.append(to_svg_line(pts_prom, args.color_promoted, 2.0))
        svg.append(to_svg_line(pts_demo, args.color_demoted, 2.0))
        svg.append(f"<text x='{x1+10}' y='{y1+15}' font-size='12'>promoted/demoted</text>\n")
    else:
        # Single panel: overlay all
        pts_prom = scale_points(steps, promoted, x0, y0, ww, hh)
        pts_demo = scale_points(steps, demoted, x0, y0, ww, hh)
        svg.append(to_svg_line(pts_prom, args.color_promoted, 2.0))
        svg.append(to_svg_line(pts_demo, args.color_demoted, 2.0))
        svg.append(f"<text x='{x0+10}' y='{y0+30}' font-size='12'>promoted/demoted</text>\n")

    svg.append("</svg>\n")

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(svg)


if __name__ == "__main__":
    main()
