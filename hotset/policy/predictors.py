"""Predictors return an expected call *count*, not a rank: break-even needs a number."""

from __future__ import annotations

from collections import defaultdict


class LRUK:
    """Recency and frequency in one number, via the LRU-K backward K-distance.

    K hits in the last D turns reads as a rate of K/D. Fewer than K sightings are
    damped, since one call is weak evidence of a rate.
    """

    name = "lru-k"

    def __init__(self, k: int = 2) -> None:
        self.k = k
        self.uses: dict[str, list[int]] = defaultdict(list)
        self.turn = 0

    def advance(self) -> None:
        """One turn elapsed. Called even on turns with no tool call, so rates decay."""
        self.turn += 1

    def reset(self) -> None:
        """Session boundary. LRU-K keeps its history: it models one continuous deployment."""

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


class Rate:
    """Decayed call rate over the whole observation window, shrunk toward zero.

    LRU-K divides by the shortest window holding k hits, a denominator selected to be
    small; dividing by every turn watched removes that selection bias. `prior` stops a
    first call from implying a rate, and decay stops an old burst from holding a schema.
    """

    name = "rate"

    def __init__(self, half_life: float = 50.0, prior: float = 25.0) -> None:
        self.gamma = 0.5 ** (1.0 / half_life)
        self.prior = prior
        self.counts: dict[str, float] = defaultdict(float)
        self.window = 0.0  # decayed turns observed, the honest denominator

    def advance(self) -> None:
        """Decay everything one turn, including turns where nothing was called."""
        self.window = self.window * self.gamma + 1.0
        for name in self.counts:
            self.counts[name] *= self.gamma

    def reset(self) -> None:
        """One continuous deployment, same as LRU-K: rates survive session boundaries."""

    def observe(self, tool: str) -> None:
        if tool:
            self.counts[tool] += 1.0

    def expected_uses(self, tool: str, horizon: int) -> float:
        """Horizon times a rate that is a real fraction of observed turns."""
        return horizon * self.counts.get(tool, 0.0) / (self.window + self.prior)

    def ranked(self, names: list[str], horizon: int) -> list[tuple[str, float]]:
        scored = [(n, self.expected_uses(n, horizon)) for n in names]
        return sorted(scored, key=lambda p: (-p[1], p[0]))


class Markov:
    """First-order transitions blended into the marginal rate as context decays.

    LRU-K knows a tool is hot but not what follows what. An edge only predicts the very
    next turn, so the horizon sum weights it by `decay**t` as the chain mixes back.
    """

    name = "markov"

    def __init__(self, alpha: float = 1.0, decay: float = 0.5) -> None:
        self.alpha = alpha  # shrinkage toward the marginal; one pseudo-count
        self.decay = decay
        self.edges: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.counts: dict[str, int] = defaultdict(int)
        self.total = 0
        self.turn = 0
        self.last = ""

    def advance(self) -> None:
        self.turn += 1

    def reset(self) -> None:
        """Session boundary. Clears context only: a transition must not span scenarios."""
        self.last = ""

    def observe(self, tool: str) -> None:
        if not tool:
            return
        if self.last:
            self.edges[self.last][tool] += 1
        self.counts[tool] += 1
        self.total += 1
        self.last = tool

    def _marginal(self, tool: str) -> float:
        return self.counts.get(tool, 0) / self.total if self.total else 0.0

    def _transition(self, tool: str) -> float:
        """P(next = tool | last), shrunk toward the marginal by `alpha` pseudo-counts."""
        marginal = self._marginal(tool)
        if not self.last:
            return marginal
        out = self.edges.get(self.last, {})
        seen = sum(out.values())
        return (out.get(tool, 0) + self.alpha * marginal) / (seen + self.alpha)

    def expected_uses(self, tool: str, horizon: int) -> float:
        """Sum of per-turn probabilities, transition-weighted near term."""
        near, far = self._transition(tool), self._marginal(tool)
        weights = sum(self.decay**t for t in range(1, horizon + 1))
        return near * weights + far * (horizon - weights)

    def ranked(self, names: list[str], horizon: int) -> list[tuple[str, float]]:
        scored = [(n, self.expected_uses(n, horizon)) for n in names]
        return sorted(scored, key=lambda p: (-p[1], p[0]))


class Ensemble:
    """Mean of member estimates. Averaging counts is only meaningful because every
    predictor returns expected uses on the same scale; a rank ensemble could not."""

    def __init__(self, members: list, name: str = "ensemble") -> None:
        self.members = members
        self.name = name

    def advance(self) -> None:
        for m in self.members:
            m.advance()

    def reset(self) -> None:
        for m in self.members:
            m.reset()

    def observe(self, tool: str) -> None:
        for m in self.members:
            m.observe(tool)

    def expected_uses(self, tool: str, horizon: int) -> float:
        return sum(m.expected_uses(tool, horizon) for m in self.members) / len(self.members)

    def ranked(self, names: list[str], horizon: int) -> list[tuple[str, float]]:
        scored = [(n, self.expected_uses(n, horizon)) for n in names]
        return sorted(scored, key=lambda p: (-p[1], p[0]))


def warm(predictor, sequences) -> None:
    """Replay historical tool sequences so a predictor starts with a prior.

    Takes bare name sequences rather than Sessions, to keep predictors independent
    of the eval package.
    """
    for sequence in sequences:
        predictor.reset()  # sequences are separate scenarios, not one long trace
        for name in sequence:
            predictor.advance()
            predictor.observe(name)


class Oracle:
    """Upper bound: counts the future directly. Separates a weak predictor from a
    workload with nothing to admit -- both otherwise show up as an empty hot set.
    """

    name = "oracle"

    def __init__(self, future: list[str]) -> None:
        self.future = future  # the whole eval trace, flattened, in order
        # Starts before the trace: plan() advances once *before* the current turn is
        # served, so after that advance the cursor must sit on the turn being planned.
        self.cursor = -1

    def advance(self) -> None:
        self.cursor += 1

    def reset(self) -> None:
        """Cursor is a position in one flat trace, so a session boundary changes nothing."""

    def observe(self, tool: str) -> None:
        """Nothing to learn: the future is already known."""

    def expected_uses(self, tool: str, horizon: int) -> float:
        start = max(self.cursor, 0)
        window = self.future[start : start + horizon]
        return float(window.count(tool))

    def ranked(self, names: list[str], horizon: int) -> list[tuple[str, float]]:
        scored = [(n, self.expected_uses(n, horizon)) for n in names]
        return sorted(scored, key=lambda p: (-p[1], p[0]))
