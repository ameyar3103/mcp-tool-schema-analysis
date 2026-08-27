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
        epoch: int = 1,
        budget: int = 0,
    ) -> None:
        self.spec = spec
        self.predictor = predictor
        self.horizon = horizon
        self.tail_k = tail_k
        # Hysteresis band: evicting the moment a tool dips below n* makes the hot set
        # flap, and every flap rewrites layer B.
        self.evict_ratio = evict_ratio
        self.max_hot = max_hot
        # Admissions batched into epochs share one prefix rewrite. Deciding every turn
        # is what made layer B lose: each membership change re-warms the prefix, so N
        # scattered admissions cost N invalidations to buy the same N schemas.
        self.epoch = epoch
        # Break-even prices tokens only, assuming a tail-load substitutes perfectly for a
        # cached schema. Measured, it does not: a stable hot set beats bare names by 5.2
        # points. `budget` buys that accuracy outright -- top-N by predicted rate, whether
        # or not they clear n* -- while the epoch keeps the prefix from churning.
        self.budget = budget
        self.turn = 0
        self.hot: list[Tool] = []
        self.rewrites = 0  # membership changes, the quantity the bill actually tracks
        self._bm25: BM25 | None = None
        self._head_tokens: tuple[int, int] | None = None
        # A schema's token count never changes, but it is re-read for every candidate
        # on every turn: 300 tools x 310 turns is 93k tokenizer calls without this.
        self._schema_tokens: dict[str, int] = {}

    def _head(self, catalog: list[Tool]) -> int:
        """Tokens above layer B. Catalog-sized, so measure it once per catalog."""
        if self._head_tokens is None or self._head_tokens[0] != len(catalog):
            plan = Plan(index=catalog)
            text = preamble(plan) + "\n\n" + layer_a(catalog)
            self._head_tokens = (len(catalog), estimate(text, self.spec.token_scale))
        return self._head_tokens[1]

    def _schema(self, tool: Tool) -> int:
        """Token cost of admitting this tool, memoized per catalog entry."""
        if tool.name not in self._schema_tokens:
            self._schema_tokens[tool.name] = estimate(layer_b([tool]), self.spec.token_scale)
        return self._schema_tokens[tool.name]

    def _threshold(self, tool: Tool, segment: int) -> float:
        """Break-even for this specific tool: bigger schemas earn admission sooner."""
        return break_even(self.spec, self._schema(tool), segment, self.horizon)

    def observe(self, tool: str) -> None:
        """What the agent actually called, which is all a deployment can see."""
        self.predictor.observe(tool)

    def reset(self) -> None:
        """New scenario. The hot set persists across sessions; only context clears."""
        self.predictor.reset()

    def plan(self, catalog: list[Tool], history: list[dict], query: str) -> Plan:
        self.predictor.advance()
        self.turn += 1
        if self._bm25 is None or self._bm25.tools != catalog:
            self._bm25 = BM25(catalog)
        # Off-epoch turns still get a tail; only membership is frozen, so the prefix
        # holds and the controller stays responsive to the current query.
        if self.turn % self.epoch:
            return self._assemble(catalog, query)

        hot_tokens = estimate(layer_b(self.hot), self.spec.token_scale) if self.hot else 0
        # Priced against the current segment: admitting changes it, but using the
        # post-admission size would make the decision depend on its own outcome.
        segment = rewritten_segment(self.spec, self._head(catalog), hot_tokens)
        scores = dict(self.predictor.ranked([t.name for t in catalog], self.horizon))

        if self.budget:
            top = sorted(catalog, key=lambda t: (-scores.get(t.name, 0.0), t.name))
            return self._commit(top[: self.budget], catalog, query)

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
        return self._commit(keep, catalog, query)

    def _commit(self, keep: list[Tool], catalog: list[Tool], query: str) -> Plan:
        """Sorted for byte-stability, then counted: a membership change is the unit billed."""
        chosen = sorted(keep, key=lambda t: t.name)
        if [t.name for t in chosen] != [t.name for t in self.hot]:
            self.rewrites += 1
        self.hot = chosen
        return self._assemble(catalog, query)

    def _assemble(self, catalog: list[Tool], query: str) -> Plan:
        """Layers A and B as they stand, plus this turn's tail. No admission decision."""
        admitted = {t.name for t in self.hot}
        if not self.tail_k:
            return Plan(index=catalog, hot=self.hot)
        tail = [
            t
            for t in self._bm25.top_k(query, self.tail_k + len(admitted))
            if t.name not in admitted
        ]
        return Plan(index=catalog, hot=self.hot, tail=tail[: self.tail_k])
