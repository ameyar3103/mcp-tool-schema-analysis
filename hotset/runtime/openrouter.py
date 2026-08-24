"""Minimal OpenRouter client. Thin on purpose: the experiment needs the exact prompt bytes."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from pydantic import BaseModel

from hotset.config import ModelSpec

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def api_key() -> str:
    """Environment first, then the gitignored .env at repo root."""
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return key
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY" and value.strip():
                return value.strip()
    raise RuntimeError(f"OPENROUTER_API_KEY not set; add it to {_ENV_FILE}")


class CacheUsage(BaseModel):
    """Per-request token accounting, normalized across OpenRouter's provider variations."""

    uncached_tokens: int  # billed at full input rate
    cached_tokens: int  # served from cache, billed at the read rate
    write_tokens: int  # committed to cache, billed at the write premium
    output_tokens: int
    reported_cost_usd: float = 0.0  # OpenRouter's own bill, for cross-checking cost_usd

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

    def total_cost_usd(self, spec: ModelSpec) -> float:
        """Prompt plus completion. This is the cost-per-task figure."""
        return self.cost_usd(spec) + self.output_tokens * spec.completion / 1e6


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
        reported_cost_usd=float(usage.get("cost") or 0.0),
    )


# Transient: worth retrying. Anything else is our bug and should surface immediately.
_RETRY = {408, 429, 500, 502, 503, 529}


def post(body: dict, attempts: int = 5) -> dict:
    """Raw Chat Completions call. Probes need byte-exact control over the body."""
    req = urllib.request.Request(
        _ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            # urllib discards the body, which is the only place the real reason lives.
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code not in _RETRY or attempt == attempts - 1:
                raise urllib.error.HTTPError(
                    exc.url, exc.code, f"{exc.reason}: {detail}", exc.headers, None
                ) from None
            time.sleep(2**attempt * 0.5 + random.random() * 0.4)
    raise RuntimeError("unreachable")


def pinned_body(spec: ModelSpec, **overrides) -> dict:
    """Request skeleton with the provider pin and reasoning already settled."""
    return {
        "model": spec.slug,
        "max_tokens": 16,
        "provider": {"only": [spec.provider], "allow_fallbacks": False},
        "usage": {"include": True},
        "reasoning": {"enabled": False},
        **overrides,
    }


def complete(
    spec: ModelSpec,
    messages: list[dict],
    tools: list[dict] | None = None,
    session_id: str | None = None,
    max_tokens: int = 256,
    reasoning: bool = False,
) -> dict:
    """One Chat Completions call, hard-pinned to a single provider so the cache stays deterministic."""
    body = {
        "model": spec.slug,
        "messages": messages,
        "max_tokens": max_tokens,
        # Pinning trades away failover for a cache that cannot silently move backends.
        "provider": {"only": [spec.provider], "allow_fallbacks": False},
        "usage": {"include": True},
        # Off by default: reasoning tokens are never cached and add pure variance to
        # cost and TTFT. Note exclude:true only hides them - it still bills them.
        "reasoning": {"enabled": reasoning},
    }
    if tools:
        body["tools"] = tools
    if session_id:
        body["session_id"] = session_id

    return post(body)
