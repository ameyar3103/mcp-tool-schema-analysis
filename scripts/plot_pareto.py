"""Pareto frontier over cost and accuracy, with dominance decided statistically.

The week-2 version of this script ranked arms on raw accuracy, which quietly asserts
that a 1.3-point difference is real. At this suite size it is not: the minimum
detectable gap is around 4 points, so most "wins" are noise wearing a decimal point.

Here an arm is dominated only when something cheaper is *not significantly worse* under
paired McNemar. That puts the burden of proof on the accuracy claim rather than on the
cost claim, which is the correct direction for a paper whose thesis is about cost: a
cheaper arm has to be shown to have lost something before it is dropped.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from hotset.eval.runner import TurnResult
from hotset.eval.significance import mcnemar, minimum_detectable, outcomes, wilson

RESULTS = Path(__file__).resolve().parents[1] / "results"
ALPHA = 0.05


def collect(salt: str) -> dict[str, list[TurnResult]]:
    """Every arm from one run. A shared salt is what makes them comparable."""
    rows: dict[str, list[TurnResult]] = defaultdict(list)
    for path in sorted(RESULTS.glob(f"*-{salt}.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            rows[record["arm"]].append(TurnResult(**record))
    return dict(rows)


def metrics(results: list[TurnResult]) -> dict:
    """Per-arm summary. Cost is per turn; the interval is what makes accuracy readable."""
    scored = [r for r in results if not r.error]
    n = len(scored) or 1
    correct = sum(r.correct for r in scored)
    lenient = sum(r.correct or r.twin for r in scored)
    lo, hi = wilson(correct, n)
    llo, lhi = wilson(lenient, n)
    cost = sum(r.cost_usd for r in scored) / n
    return {
        "accuracy": correct / n,
        "lenient": lenient / n,
        "lo": lo,
        "hi": hi,
        "lenient_lo": llo,
        "lenient_hi": lhi,
        "cost_per_turn": cost,
        "cost_per_correct": cost / (correct / n) if correct else float("inf"),
        "hit_rate": sum(r.hit_rate() for r in scored) / n,
        "turns": len(scored),
    }


def dominates(a: str, b: str, arms: dict, paired: dict, alpha: float = ALPHA) -> bool:
    """a beats b if it is cheaper and its accuracy is not provably lower."""
    if arms[a]["cost_per_turn"] >= arms[b]["cost_per_turn"]:
        return False
    if arms[a]["accuracy"] >= arms[b]["accuracy"]:
        return True
    _, _, p = mcnemar(paired[a], paired[b])
    return p >= alpha  # cheaper, and the accuracy gap is indistinguishable from zero


def antichain(arms: dict, paired: dict, alpha: float = ALPHA) -> set[str]:
    """Arms no single other arm dominates."""
    return {a for a in arms if not any(dominates(b, a, arms, paired, alpha) for b in arms if b != a)}


def frontier(arms: dict, paired: dict, alpha: float = ALPHA) -> set[str]:
    """The antichain, plus anything that measurably beats something in it.

    "Not significantly worse" is not transitive, so dominance chains: a cheap arm ties
    a middling one, the middling one ties an accurate one, and the accurate arm drops
    out even though it beats the survivor at p<0.01. That is a correct pairwise
    statement and a useless recommendation, so arms that buy a real accuracy gain over
    a survivor are put back until the set stops growing.
    """
    best = antichain(arms, paired, alpha)
    while True:
        add = {
            d
            for d in set(arms) - best
            for k in best
            if arms[d]["accuracy"] > arms[k]["accuracy"] and mcnemar(paired[d], paired[k])[2] < alpha
        }
        if not add:
            return best
        best |= add


def inversions(arms: dict, paired: dict, best: set[str], alpha: float = ALPHA) -> list[tuple]:
    """Dropped arms that significantly beat a surviving one.

    "Not significantly worse" is not transitive, so dominance can chain: a cheap arm
    beats a middling one, the middling one beats an accurate one, and the accurate arm
    falls off the frontier even though it beats the survivor outright. The frontier is
    still correct pairwise, but reporting it alone would hide that, so every such
    inversion is listed rather than left for a reader to rediscover.
    """
    out = []
    for d in set(arms) - best:
        for k in best:
            *_, pv = mcnemar(paired[d], paired[k])
            if pv < alpha and arms[d]["accuracy"] > arms[k]["accuracy"]:
                out.append((d, k, pv))
    return sorted(out, key=lambda t: t[2])


def naive_frontier(arms: dict) -> set[str]:
    """The week-2 rule, kept only to show what significance testing removes."""
    return {
        a
        for a in arms
        if not any(
            arms[b]["cost_per_turn"] <= arms[a]["cost_per_turn"]
            and arms[b]["accuracy"] >= arms[a]["accuracy"]
            and b != a
            for b in arms
        )
    }


def report(arms: dict, best: set[str], naive: set[str], flags: list[tuple] = (),
           readmitted: set[str] = frozenset()) -> None:
    """The numbers are the deliverable; the plot is a rendering of them."""
    turns = max(m["turns"] for m in arms.values())
    print(f"{'arm':16} {'acc':>6} {'95% CI':>15} {'lenient':>8} {'hit':>6} "
          f"{'$/turn':>10} {'$/correct':>11}  front")
    for name, m in sorted(arms.items(), key=lambda kv: kv[1]["cost_per_turn"]):
        ci = f"[{m['lo']:.1%},{m['hi']:.1%}]"
        print(f"{name:16} {m['accuracy']:6.1%} {ci:>15} {m['lenient']:8.1%} {m['hit_rate']:6.1%} "
              f"${m['cost_per_turn']:9.6f} ${m['cost_per_correct']:10.6f}  "
              f"{'*' if name in best else ''}")
    print(f"\nminimum detectable accuracy gap at n={turns}: {minimum_detectable(turns):.1%}")
    print(f"statistical frontier: {sorted(best)}")
    dropped = naive - best
    kept = best - naive
    if dropped:
        print(f"on the naive frontier only because their accuracy edge is noise: {sorted(dropped)}")
    if kept:
        print(f"retained despite a nominal loss that is not significant: {sorted(kept)}")
    if readmitted:
        print(f"re-admitted past a dominance chain by beating a survivor: {sorted(readmitted)}")
    for d, k, pv in flags:
        print(f"UNRESOLVED inversion: {d} beats {k} at p={pv:.4f} yet is excluded")


def plot(arms: dict, best: set[str], out: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; text report only)")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, m in arms.items():
        on_front = name in best
        # The bar is the Wilson interval: it shows how much of the ranking is noise.
        ax.errorbar(m["cost_per_turn"], m["accuracy"],
                    yerr=[[m["accuracy"] - m["lo"]], [m["hi"] - m["accuracy"]]],
                    fmt="o" if on_front else "x", ms=9 if on_front else 7,
                    capsize=3, lw=1, alpha=0.9 if on_front else 0.5, zorder=3)
        ax.annotate(f"  {name}", (m["cost_per_turn"], m["accuracy"]), fontsize=9,
                    fontweight="bold" if on_front else "normal", va="center")
    edge = sorted((arms[n]["cost_per_turn"], arms[n]["accuracy"]) for n in best)
    ax.plot([c for c, _ in edge], [a for _, a in edge], "--", lw=1, alpha=0.5, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("cost per turn (USD, log scale)")
    ax.set_ylabel("tool-selection accuracy")
    ax.set_title("Tool routing Pareto frontier (bars: 95% Wilson CI)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / out, dpi=150)
    print(f"wrote {RESULTS / out}")


def main(salt: str, out: str = "pareto.png", lenient: bool = False) -> None:
    raw = collect(salt)
    if not raw:
        raise SystemExit(f"no results for salt {salt}")
    arms = {name: metrics(rs) for name, rs in raw.items()}
    if lenient:
        # Same frontier machinery, scoring a synthetic twin as a hit. The interval has
        # to move with the point estimate or the plotted bars go negative.
        arms = {
            n: dict(m, accuracy=m["lenient"], lo=m["lenient_lo"], hi=m["lenient_hi"])
            for n, m in arms.items()
        }
    paired = {name: outcomes(rs, lenient) for name, rs in raw.items()}
    best = frontier(arms, paired)
    readmitted = best - antichain(arms, paired)
    print(f"scoring: {'lenient (twin counts as correct)' if lenient else 'strict'}")
    report(arms, best, naive_frontier(arms), inversions(arms, paired, best), readmitted)
    plot(arms, best, out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--lenient"]
    main(*args, lenient="--lenient" in sys.argv)
