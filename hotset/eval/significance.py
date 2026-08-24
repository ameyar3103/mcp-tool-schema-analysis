"""Paired significance tests over turn-level results.

Arms are evaluated on identical turns, so the pairing is real and McNemar is the
right test: it conditions on the discordant pairs and ignores the turns every arm
gets right, which are most of them. An unpaired proportion test throws that away
and needs several times the sample for the same power.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import comb

from hotset.eval.runner import TurnResult

_Z = 1.959964  # two-sided 95%


@dataclass(frozen=True)
class Comparison:
    """Discordant counts and an exact p-value for one pair of arms."""

    a: str
    b: str
    a_only: int  # a correct, b wrong
    b_only: int
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def outcomes(results: list[TurnResult], lenient: bool = False) -> dict[tuple[int, int], bool]:
    """Turn key to correctness. Transport errors are dropped, not scored as wrong.

    Under `lenient`, picking a synthetic twin counts as correct. The two views answer
    different questions: strict asks whether the arm found the labeled tool, lenient
    asks whether it found a tool that does the job. An arm holding 225 near-duplicates
    in context loses the strict comparison for a reason that has nothing to do with
    routing, so a strict-only result cannot separate the two explanations.
    """
    return {(r.session, r.turn): (r.correct or (lenient and r.twin)) for r in results if not r.error}


def wilson(correct: int, total: int) -> tuple[float, float]:
    """Wilson interval. The normal approximation misbehaves near the 0/1 ceiling."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    d = 1 + _Z**2 / total
    centre = (p + _Z**2 / (2 * total)) / d
    half = _Z * math.sqrt(p * (1 - p) / total + _Z**2 / (4 * total**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(a: dict[tuple[int, int], bool], b: dict[tuple[int, int], bool]) -> tuple[int, int, float]:
    """Exact binomial McNemar. Exact rather than chi-square: discordant counts here are small."""
    shared = a.keys() & b.keys()
    a_only = sum(1 for k in shared if a[k] and not b[k])
    b_only = sum(1 for k in shared if b[k] and not a[k])
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    tail = sum(comb(n, i) for i in range(min(a_only, b_only) + 1))
    return a_only, b_only, min(1.0, 2 * tail / 2**n)


def compare(arms: dict[str, list[TurnResult]], lenient: bool = False) -> list[Comparison]:
    """Every pair, ordered by accuracy so the strongest arm leads each comparison."""
    scored = {name: outcomes(rs, lenient) for name, rs in arms.items()}
    order = sorted(scored, key=lambda n: -sum(scored[n].values()))
    out = []
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            x, y, p = mcnemar(scored[a], scored[b])
            out.append(Comparison(a, b, x, y, p))
    return out


def minimum_detectable(n: int, discordant_rate: float = 0.25) -> float:
    """Accuracy gap detectable at 80% power, as a fraction. Sizes future suites.

    Under McNemar, power depends on the discordant count, not the sample size, so a
    suite of mostly-agreed turns is far weaker than its turn count suggests.
    """
    d = max(1.0, n * discordant_rate)
    return (_Z + 0.8416) / (2 * math.sqrt(d)) * (d / n)
