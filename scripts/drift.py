"""Did behaviour degrade mid-run, or only get cheaper?

An arm's headline accuracy is an average over every turn of every session, which hides
two failure modes a cost paper has to rule out. First, decay across the deployment: an
adaptive policy that admits the wrong tools gets worse as it goes. Second, decay within
a conversation: a compact prompt may hold up on turn 0 and fall apart on turn 4, once
history has grown and the schema the model needs is no longer nearby.

Both are invisible in the mean and obvious in the spans.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from hotset.eval.significance import fisher, wilson
from hotset.eval.spans import drift, load

TRACES = Path(__file__).resolve().parents[1] / "traces"


def turns(spans) -> list:
    return [s for s in spans if s.name == "turn"]


def by_position(spans) -> dict[int, tuple[int, int]]:
    """Accuracy against turn index, which is also prompt length and history depth."""
    buckets: dict[int, list[bool]] = defaultdict(list)
    for s in turns(spans):
        buckets[s.attributes.get("hotset.turn", 0)].append(bool(s.attributes.get("hotset.correct")))
    return {k: (sum(v), len(v)) for k, v in sorted(buckets.items())}


def halves(spans) -> tuple[tuple[int, int], tuple[int, int]]:
    """First versus last half of the deployment, in wall-clock order."""
    flags = [bool(s.attributes.get("hotset.correct")) for s in sorted(turns(spans), key=lambda s: s.start_ms)]
    mid = len(flags) // 2
    return (sum(flags[:mid]), mid), (sum(flags[mid:]), len(flags) - mid)


def relative(spans, reference) -> tuple[tuple[int, int], tuple[int, int], float]:
    """Has this arm's loss rate against a reference changed between halves?

    Comparing an arm's own halves cannot separate "the agent degraded" from "the second
    half of the suite is easier" — every arm replays the same sessions in the same
    order, so scenario difficulty moves all of them together. What is left after
    differencing against a reference arm is drift specific to this arm.
    """
    keys = sorted(set(spans) & set(reference))
    mid = len(keys) // 2

    def losses(window):
        """Turns the reference got right and this arm did not, out of the window."""
        return sum(reference[k] and not spans[k] for k in window), len(window)

    (l1, n1), (l2, n2) = losses(keys[:mid]), losses(keys[mid:])
    # Rate of loss, not ratio of wins to losses: an arm that agrees perfectly in the
    # first half has no ratio to compare, but its loss rate is still a well-defined 0.
    return (l1, n1), (l2, n2), fisher(l1, n1 - l1, l2, n2 - l2)


def flags(spans) -> dict:
    """Turn spans keyed by (session, turn), so two arms can be paired on the same turn."""
    return {
        (s.attributes.get("hotset.session"), s.attributes.get("hotset.turn")): bool(
            s.attributes.get("hotset.correct")
        )
        for s in turns(spans)
    }


def report(arm: str, spans) -> None:
    n = len(turns(spans))
    if not n:
        print(f"{arm:16} no turn spans")
        return
    (a, an), (b, bn) = halves(spans)
    alo, ahi = wilson(a, an or 1)
    blo, bhi = wilson(b, bn or 1)
    # Raw halves are reported for context only; the verdict comes from relative(), since
    # a shift shared by every arm is the suite changing rather than the agent.
    print(f"{arm:16} n={n:4d}  first-half {a / max(an, 1):5.1%} [{alo:.1%},{ahi:.1%}]  "
          f"second-half {b / max(bn, 1):5.1%} [{blo:.1%},{bhi:.1%}]")
    positions = by_position(spans)
    line = "  ".join(f"t{k}:{c / max(t, 1):.0%}({t})" for k, (c, t) in positions.items())
    print(f"{'':16} by position   {line}")
    series = drift(spans, window=20)
    if series:
        print(f"{'':16} rolling(20)   min {min(series):.0%}  max {max(series):.0%}")


def main(salt: str, reference: str = "full-catalog") -> None:
    paths = sorted(TRACES.glob(f"*-{salt}.jsonl"))
    if not paths:
        raise SystemExit(f"no traces for salt {salt} in {TRACES}")
    arms = {}
    for path in paths:
        spans = load(path)
        arms[spans[0].attributes.get("hotset.arm", path.stem)] = spans
    for name, spans in arms.items():
        report(name, spans)
    ref = arms.get(reference) or arms[next(iter(arms))]
    reference = next(n for n, v in arms.items() if v is ref)
    print(f"\nregression against {reference} (turns lost per half, Fisher exact)")
    for name, spans in arms.items():
        if name == reference:
            continue
        first, second, p = relative(flags(spans), flags(ref))
        verdict = "DRIFT" if p < 0.05 else "flat"
        print(f"{name:16} lost {first[0]:3}/{first[1]:<3} then {second[0]:3}/{second[1]:<3}"
              f"  p={p:.3f}  {verdict}")


if __name__ == "__main__":
    main(*sys.argv[1:])
