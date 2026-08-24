"""The HotSet policy: layer A always, layer B by admission economics, layer C per turn."""

from __future__ import annotations

from hotset.config import ModelSpec
from hotset.corpus.models import Tool
from hotset.layout.prompt import layer_a, layer_b, preamble
from hotset.layout.tokens import estimate
from hotset.policy.base import Plan
from hotset.policy.economics import break_even, rewritten_segment
from hotset.policy.retrieval import BM25


class HotSet:
    """Cache-aware admission. Every tool stays nameable; only earners get a schema."""

    name = "hotset"
    # One shared prefix serves every session, so the predictor is shared state and
    # sessions must run in order or worker threads race on the turn counter.
    stateful = True

    def __init__(
        self,
        spec: ModelSpec,
        predictor,
        horizon: int = 10,
        tail_k: int = 3,
        evict_ratio: float = 0.5,
        max_hot: int = 32,
    ) -> None:
        self.spec = spec
        self.predictor = predictor
        self.horizon = horizon
        self.tail_k = tail_k
        # Hysteresis band: evicting the moment a tool dips below n* makes the hot set
        # flap, and every flap rewrites layer B.
        self.evict_ratio = evict_ratio
        self.max_hot = max_hot
        self.hot: list[Tool] = []
        self._bm25: BM25 | None = None
        self._head_tokens: tuple[int, int] | None = None

    def _head(self, catalog: list[Tool]) -> int:
        """Tokens above layer B. Catalog-sized, so measure it once per catalog."""
        if self._head_tokens is None or self._head_tokens[0] != len(catalog):
            plan = Plan(index=catalog)
            text = preamble(plan) + "\n\n" + layer_a(catalog)
            self._head_tokens = (len(catalog), estimate(text, self.spec.token_scale))
        return self._head_tokens[1]

    def _threshold(self, tool: Tool, segment: int) -> float:
        """Break-even for this specific tool: bigger schemas earn admission sooner."""
        schema = estimate(layer_b([tool]), self.spec.token_scale)
        return break_even(self.spec, schema, segment, self.horizon)

    def observe(self, tool: str) -> None:
        """What the agent actually called, which is all a deployment can see."""
        self.predictor.observe(tool)

    def reset(self) -> None:
        """New scenario. The hot set persists across sessions; only context clears."""
        self.predictor.reset()

    def plan(self, catalog: list[Tool], history: list[dict], query: str) -> Plan:
        self.predictor.advance()
        if self._bm25 is None or self._bm25.tools != catalog:
            self._bm25 = BM25(catalog)

        hot_tokens = estimate(layer_b(self.hot), self.spec.token_scale) if self.hot else 0
        # Priced against the current segment: admitting changes it, but using the
        # post-admission size would make the decision depend on its own outcome.
        segment = rewritten_segment(self.spec, self._head(catalog), hot_tokens)
        scores = dict(self.predictor.ranked([t.name for t in catalog], self.horizon))

        keep = [
            t
            for t in self.hot
            if scores.get(t.name, 0.0) >= self._threshold(t, segment) * self.evict_ratio
        ]
        held = {t.name for t in keep}
        candidates = sorted(
            (t for t in catalog if t.name not in held),
            key=lambda t: (-scores.get(t.name, 0.0), t.name),
        )
        for tool in candidates:
            if len(keep) >= self.max_hot:
                break
            if scores.get(tool.name, 0.0) >= self._threshold(tool, segment):
                keep.append(tool)
        self.hot = sorted(keep, key=lambda t: t.name)

        admitted = {t.name for t in self.hot}
        tail = [
            t
            for t in self._bm25.top_k(query, self.tail_k + len(admitted))
            if t.name not in admitted
        ]
        return Plan(index=catalog, hot=self.hot, tail=tail[: self.tail_k])
