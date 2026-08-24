"""Lexical retrieval over the catalog — the ranker behind the RAG-over-tools baseline."""

from __future__ import annotations

import math
import re
from collections import Counter

from hotset.corpus.models import Tool

_TOKEN = re.compile(r"[a-z0-9]+")


def terms(text: str) -> list[str]:
    """Split on camel and snake boundaries so `get_current_time` matches a query for `time`."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return _TOKEN.findall(text.lower().replace("_", " ").replace(".", " "))


def _document(tool: Tool) -> list[str]:
    """Name and description carry the signal; arg names disambiguate near-duplicates."""
    return terms(f"{tool.name} {tool.description} {' '.join(tool.arg_names)}")


class BM25:
    """Okapi BM25. Dependency-free and deterministic, so the baseline is reproducible."""

    def __init__(self, catalog: list[Tool], k1: float = 1.5, b: float = 0.75) -> None:
        self.tools = list(catalog)
        self.k1, self.b = k1, b
        docs = [_document(t) for t in self.tools]
        self.freqs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg_len = sum(self.lengths) / max(len(docs), 1)
        df = Counter(term for d in docs for term in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        """BM25 score of every catalog tool against the query."""
        q = terms(query)
        out = []
        for freq, length in zip(self.freqs, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / max(self.avg_len, 1e-9))
            out.append(
                sum(
                    self.idf.get(t, 0.0) * freq[t] * (self.k1 + 1) / (freq[t] + norm)
                    for t in q
                    if freq[t]
                )
            )
        return out

    def top_k(self, query: str, k: int) -> list[Tool]:
        """Highest-scoring k tools. Ties break on name so results never depend on catalog order."""
        ranked = sorted(
            zip(self.scores(query), self.tools, strict=True), key=lambda p: (-p[0], p[1].name)
        )
        return [t for _, t in ranked[:k]]
