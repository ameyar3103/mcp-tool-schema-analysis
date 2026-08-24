"""The comparison arms. Each fails differently, which is why all three are needed."""

from __future__ import annotations

from hotset.corpus.models import Tool
from hotset.layout.serialize import canonical_tool
from hotset.policy.base import Plan
from hotset.policy.retrieval import BM25


class FullCatalog:
    """Baseline 1: every schema in the prefix. Perfect cache, enormous constant."""

    name = "full-catalog"

    def plan(self, catalog: list[Tool], history: list[dict], query: str) -> Plan:
        return Plan(hot=list(catalog))


class RagOverTools:
    """Baseline 2: retrieve top-k per turn and rebuild the prefix.

    The retrieved set changes with the query, so the cached prefix is invalidated on
    almost every turn. Cheap per turn in tokens, expensive because none of them cache.
    """

    name = "rag-over-tools"

    def __init__(self, k: int = 8) -> None:
        self.k = k
        self._bm25: BM25 | None = None

    def plan(self, catalog: list[Tool], history: list[dict], query: str) -> Plan:
        if self._bm25 is None or self._bm25.tools != catalog:
            self._bm25 = BM25(catalog)
        return Plan(hot=self._bm25.top_k(query, self.k))


_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_tools",
        "description": "Search the tool registry. Returns full schemas for matching tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Capability you need."},
                "limit": {"type": "integer", "description": "Max results, default 5."},
            },
            "required": ["query"],
        },
    },
}

_LAZY_HINT = """You do not know any tool names yet, and no schemas are loaded.

Always call `search_tools` first to discover what exists. It returns full schemas.
Only after a search may you call one of the tools it returned, by its exact name.
Never guess a tool name: a guessed name does not exist and the call will fail.
You may search more than once if the first results do not fit."""


class LazyDiscovery:
    """Baseline 3: MCP-Zero style. Tiny prefix, paid for in extra round trips."""

    name = "lazy-discovery"

    def __init__(self, limit: int = 5) -> None:
        self.limit = limit
        self._bm25: BM25 | None = None

    def plan(self, catalog: list[Tool], history: list[dict], query: str) -> Plan:
        # The dispatcher would claim to be the only way to call a tool, which
        # competes with search_tools. search_tools is this arm's format primer.
        return Plan(extra_tools=[_SEARCH_TOOL], instructions=_LAZY_HINT, use_dispatcher=False)

    def serves(self, name: str) -> bool:
        """Does this policy handle the named tool itself, rather than the environment?"""
        return name == "search_tools"

    def serve(self, catalog: list[Tool], args: dict) -> str:
        """Registry lookup. Returns schemas, so a hit costs a full uncached schema."""
        if self._bm25 is None or self._bm25.tools != catalog:
            self._bm25 = BM25(catalog)
        limit = min(int(args.get("limit") or self.limit), 10)
        hits = self._bm25.top_k(str(args.get("query", "")), limit)
        return "\n".join(canonical_tool(t) for t in hits) or "No matching tools."


class StaticHotSet:
    """Baseline 4: index plus a frequency-ranked hot set, fixed for the whole run."""

    name = "static-hot-set"

    def __init__(self, hot: list[Tool]) -> None:
        self.hot = list(hot)

    def plan(self, catalog: list[Tool], history: list[dict], query: str) -> Plan:
        return Plan(index=list(catalog), hot=self.hot)
