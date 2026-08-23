"""Model registry. Prices drive the break-even threshold, so they are data, not constants."""

from __future__ import annotations

from pydantic import BaseModel


class ModelSpec(BaseModel):
    """Pricing and cache semantics for one OpenRouter-routed model, USD per MTok."""

    slug: str  # OpenRouter model id
    provider: str  # endpoint tag for provider.only, pins the cache to one backend
    uncached: float  # u in the break-even derivation
    cached: float  # c
    write: float  # cache-write premium, charged once per admission
    min_cacheable_tokens: int  # below this the provider silently skips caching
    context_tokens: int

    @property
    def cost_ratio(self) -> float:
        """u/c. Higher means promotion pays off sooner."""
        return self.uncached / self.cached


MODELS: dict[str, ModelSpec] = {
    # Sweep workhorse: ~33x cheaper than Sonnet, 1M context, single endpoint so no drift.
    "qwen-flash": ModelSpec(
        slug="qwen/qwen3.7-flash",
        provider="alibaba",
        uncached=0.030,
        cached=0.006,
        write=0.038,
        min_cacheable_tokens=1024,  # unverified; probe 1 measures the real floor
        context_tokens=1_000_000,
    ),
    # Headline run: u/c = 10 reproduces the arithmetic in the design doc.
    "haiku": ModelSpec(
        slug="anthropic/claude-haiku-4.5",
        provider="anthropic",
        uncached=1.00,
        cached=0.10,
        write=1.25,
        min_cacheable_tokens=4096,
        context_tokens=200_000,
    ),
    "sonnet": ModelSpec(
        slug="anthropic/claude-sonnet-5",
        provider="anthropic",
        uncached=2.00,
        cached=0.20,
        write=2.50,
        min_cacheable_tokens=1024,
        context_tokens=1_000_000,
    ),
}
