"""Minimal OpenRouter client. Thin on purpose: the experiment needs the exact prompt bytes."""

from __future__ import annotations

import json
import os
import urllib.request

from pydantic import BaseModel

from hotset.config import ModelSpec

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class CacheUsage(BaseModel):
    """Per-request token accounting, normalized across OpenRouter's provider variations."""

    uncached_tokens: int  # billed at full input rate
    cached_tokens: int  # served from cache, billed at the read rate
    write_tokens: int  # committed to cache, billed at the write premium
    output_tokens: int

    @property
    def prompt_tokens(self) -> int:
        """Total prompt size; the single fields each report only one slice of it."""
        return self.uncached_tokens + self.cached_tokens + self.write_tokens

    def hit_rate(self) -> float:
        """Fraction of this request's prompt served from cache."""
        # Writes stay in the denominator: a cold start genuinely is a 0% hit.
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def cost_usd(self, spec: ModelSpec) -> float:
        """Prompt-side cost in USD, pricing each token slice at its own rate."""
        return (
            self.uncached_tokens * spec.uncached
            + self.cached_tokens * spec.cached
            + self.write_tokens * spec.write
        ) / 1e6  # registry prices are per MTok


def parse_usage(payload: dict) -> CacheUsage:
    """Pull cache accounting out of a Chat Completions response, tolerating absent fields."""
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    write = int(details.get("cache_write_tokens") or 0)
    # OpenRouter reports prompt_tokens as the total; the cached slices are carved out of it.
    total = int(usage.get("prompt_tokens") or 0)
    return CacheUsage(
        uncached_tokens=max(total - cached - write, 0),
        cached_tokens=cached,
        write_tokens=write,
        output_tokens=int(usage.get("completion_tokens") or 0),
    )


def complete(spec: ModelSpec, messages: list[dict], tools: list[dict] | None = None,
             session_id: str | None = None, max_tokens: int = 256) -> dict:
    """One Chat Completions call, hard-pinned to a single provider so the cache stays deterministic."""
    body = {
        "model": spec.slug,
        "messages": messages,
        "max_tokens": max_tokens,
        # Pinning trades away failover for a cache that cannot silently move backends.
        "provider": {"only": [spec.provider], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    if tools:
        body["tools"] = tools
    if session_id:
        body["session_id"] = session_id

    req = urllib.request.Request(
        _ENDPOINT,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())
