"""Report accuracy with intervals and paired tests for one sweep.

Usage: significance.py <salt>

The salt alone identifies a run: it is what guarantees no arm inherited a warm prefix
from another, so arms sharing one are exactly the arms that are comparable.
"""

from __future__ import annotations

import sys

from hotset.eval.runner import RESULTS, load
from hotset.eval.significance import compare, minimum_detectable, outcomes, wilson


def collect(salt: str) -> dict[str, list]:
    """Keyed on the arm recorded inside each file, not on parsing the filename."""
    out = {}
    for path in sorted(RESULTS.glob(f"*-{salt}.jsonl")):
        rows = load(path)
        if rows:
            out[rows[0].arm] = rows
    return out


def main() -> None:
    salt = sys.argv[1]
    arms = collect(salt)
    if not arms:
        raise SystemExit(f"no results for salt {salt}")
    scored = {name: outcomes(rs) for name, rs in arms.items()}
    n = max(len(v) for v in scored.values())
    model = next(iter(arms.values()))[0].model
    print(f"{model} | {n} scored turns | salt {salt}\n")
    for name in sorted(scored, key=lambda k: -sum(scored[k].values())):
        k, total = sum(scored[name].values()), len(scored[name])
        lo, hi = wilson(k, total)
        print(f"{name:16} {k:3}/{total:<3} {k / total:6.1%}  95% CI [{lo:5.1%}, {hi:5.1%}]")
    print()
    for c in compare(arms):
        mark = "  <-- significant" if c.significant else ""
        print(f"{c.a:16} vs {c.b:16} {c.a_only:3}/{c.b_only:<3}  p={c.p_value:.3f}{mark}")
    print(f"\nsmallest gap detectable at 80% power, n={n}: {minimum_detectable(n):.1%}")


if __name__ == "__main__":
    main()
