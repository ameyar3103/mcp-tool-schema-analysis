"""Admission economics: when is caching a schema cheaper than tail-loading it?

Derived from the billing model rather than assumed. Over a horizon of T turns, for a
tool whose schema is P tokens sitting in a cache segment of S tokens:

    admitted     = S*write + (T-1)*S*cached
    not admitted = T*(S-P)*cached + n*P*uncached

Setting admitted <= not admitted and solving for n gives break_even() below. Both
branches are validated against measured Anthropic runs in tests/test_economics.py.
"""

from __future__ import annotations

import math

from hotset.config import ModelSpec

_PER_TOKEN = 1e6  # registry prices are per million tokens


def tail_cost(spec: ModelSpec, schema_tokens: int) -> float:
    """One ephemeral tail-load: the whole schema billed uncached, on every use."""
    return schema_tokens * spec.uncached / _PER_TOKEN


def admission_cost(spec: ModelSpec, segment_tokens: int, horizon: int = 1) -> float:
    """Rewrite the segment once, then read it for the rest of the horizon."""
    write = segment_tokens * spec.write
    reads = (horizon - 1) * segment_tokens * spec.cached
    return (write + reads) / _PER_TOKEN


def rewritten_segment(spec: ModelSpec, head_tokens: int, hot_tokens: int) -> int:
    """What admission actually invalidates.

    Layer B alone where the provider honours a second breakpoint; the entire prefix
    where it does not. This single branch moves break-even by ~9.5x (see Q7).
    """
    return hot_tokens if spec.cache_breakpoints > 1 else head_tokens + hot_tokens


def break_even(spec: ModelSpec, schema_tokens: int, segment_tokens: int, horizon: int = 1) -> float:
    """Uses within the horizon at which admitting beats tail-loading."""
    if schema_tokens <= 0:
        return math.inf
    rewrite = segment_tokens * (spec.write - spec.cached)
    carry = horizon * spec.cached  # the admitted schema is re-read every remaining turn
    return rewrite / (schema_tokens * spec.uncached) + carry / spec.uncached


def asymptotic_rate(spec: ModelSpec) -> float:
    """Call rate a tool must exceed for admission to ever pay, at any schema size.

    Taking n*/T as T grows, the rewrite term vanishes and only the carrying cost
    survives: a cached token still bills at `cached`, so a schema earns its place iff
    it is called more often than the provider's cached/uncached price ratio. Schema
    size and segment size do not change this bound — they only set how long the
    deployment must run before it holds.
    """
    return spec.cached / spec.uncached


def amortization_horizon(
    spec: ModelSpec, schema_tokens: int, segment_tokens: int, rate: float
) -> float:
    """Turns of traffic before admitting a tool at `rate` pays for itself.

    Infinite when the rate sits below the asymptotic bound, which is the useful
    answer: no horizon rescues a tool that is called too rarely.
    """
    margin = rate - asymptotic_rate(spec)
    if margin <= 0 or schema_tokens <= 0:
        return math.inf
    rewrite = segment_tokens * (spec.write - spec.cached)
    return rewrite / (schema_tokens * spec.uncached) / margin
