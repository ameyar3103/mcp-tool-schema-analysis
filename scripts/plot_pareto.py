"""Pareto plot: cost per correct call against accuracy, one point per arm."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def collect(salt: str) -> dict[str, dict]:
    """Every arm from one run. A shared salt is what makes them comparable."""
    rows: dict[str, list[dict]] = defaultdict(list)
    for path in RESULTS.glob(f"*-{salt}.jsonl"):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if not record["error"]:
                rows[record["arm"]].append(record)
    out = {}
    for arm, records in rows.items():
        n = len(records)
        accuracy = sum(r["correct"] for r in records) / n
        cost = sum(r["cost_usd"] for r in records) / n
        out[arm] = {
            "accuracy": accuracy,
            "cost_per_turn": cost,
            "cost_per_correct": cost / accuracy if accuracy else float("inf"),
            "hit_rate": sum(
                r["cached"] / max(r["cached"] + r["written"] + r["uncached"], 1) for r in records
            )
            / n,
            "turns": n,
        }
    return out


def frontier(points: list[tuple[float, float, str]]) -> set[str]:
    """Arms no other arm beats on both cost and accuracy at once."""
    keep = set()
    for cost, accuracy, name in points:
        dominated = any(c <= cost and a >= accuracy and n != name for c, a, n in points)
        if not dominated:
            keep.add(name)
    return keep


def report(arms: dict, best: set[str]) -> None:
    """Text frontier. The numbers are the deliverable; the plot is a rendering of them."""
    print(f"{'arm':16} {'acc':>6} {'hit':>6} {'$/turn':>10} {'$/correct':>11}  front")
    for name, m in sorted(arms.items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"{name:16} {m['accuracy']:6.1%} {m['hit_rate']:6.1%} ${m['cost_per_turn']:9.6f} "
              f"${m['cost_per_correct']:10.6f}  {'*' if name in best else ''}")


def main(salt: str, out: str = "pareto.png") -> None:
    arms = collect(salt)
    if not arms:
        raise SystemExit(f"no results for salt {salt}")
    points = [(m["cost_per_turn"], m["accuracy"], name) for name, m in arms.items()]
    best = frontier(points)
    report(arms, best)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; text report only)")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for cost, accuracy, name in sorted(points):
        on_front = name in best
        ax.scatter(cost, accuracy, s=110 if on_front else 70,
                   marker="o" if on_front else "x", zorder=3)
        ax.annotate(f"  {name}", (cost, accuracy), fontsize=9,
                    fontweight="bold" if on_front else "normal", va="center")
    edge = sorted((c, a) for c, a, n in points if n in best)
    ax.plot([c for c, _ in edge], [a for _, a in edge], "--", lw=1, alpha=0.5, zorder=2)
    ax.set_xlabel("cost per turn (USD)")
    ax.set_ylabel("tool-selection accuracy")
    ax.set_title(f"Tool routing Pareto frontier — run {salt}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / out, dpi=150)
    print(f"wrote {RESULTS / out}\nfrontier: {sorted(best)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
