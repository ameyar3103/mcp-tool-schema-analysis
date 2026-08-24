"""Predictors: estimate how often a tool will be called over the next few turns.

They return an expected *count*, not a rank. The admission controller compares that
count against a break-even threshold, which a bare ordering cannot answer.
"""

from __future__ import annotations

from collections import defaultdict


class LRUK:
    """Recency and frequency in one number, via the LRU-K backward K-distance.

    A tool called K times within the last D turns is firing at roughly K/D per turn,
    so its expected uses over a horizon are horizon*K/D. Tools seen fewer than K
    times get a damped estimate: one sighting is weak evidence of a rate.
    """

    name = "lru-k"

    def __init__(self, k: int = 2) -> None:
        self.k = k
        self.uses: dict[str, list[int]] = defaultdict(list)
        self.turn = 0

    def advance(self) -> None:
        """One turn elapsed. Called even on turns with no tool call, so rates decay."""
        self.turn += 1

    def observe(self, tool: str) -> None:
        """Record a call at the current turn."""
        if tool:
            self.uses[tool].append(self.turn)

    def expected_uses(self, tool: str, horizon: int) -> float:
        """Predicted calls over the next `horizon` turns."""
        hits = self.uses.get(tool)
        if not hits:
            return 0.0
        k = min(self.k, len(hits))
        span = max(self.turn - hits[-k] + 1, 1)
        # Damp when we have fewer than k sightings: k/k guards against a single hit
        # at the current turn implying a rate of one call per turn.
        confidence = len(hits) / self.k if len(hits) < self.k else 1.0
        return horizon * (k / span) * confidence

    def ranked(self, names: list[str], horizon: int) -> list[tuple[str, float]]:
        """Every candidate scored, highest first, ties broken on name for determinism."""
        scored = [(n, self.expected_uses(n, horizon)) for n in names]
        return sorted(scored, key=lambda p: (-p[1], p[0]))


def warm(predictor, sequences) -> None:
    """Replay historical tool sequences so a predictor starts with a prior.

    Takes bare name sequences rather than Sessions, to keep predictors independent
    of the eval package.
    """
    for sequence in sequences:
        for name in sequence:
            predictor.advance()
            predictor.observe(name)
