"""Span-level traces, OpenTelemetry-shaped.

Turn-level accuracy answers "did the agent get the right tool", but the objection that
actually lands is "your cost went down — did behaviour degrade somewhere in the middle
of a run in a way the final metric cannot see?" That question needs the sequence, not
the aggregate, so every model call gets a span and every session gets a trace.

Attribute names follow the OTel `gen_ai.*` semantic conventions where they exist, so
these traces load into ordinary tracing tooling instead of a bespoke viewer.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, Field

TRACES = Path(__file__).resolve().parents[2] / "traces"


class Span(BaseModel):
    """One unit of work. Durations are milliseconds, matching OTel's own convention."""

    trace_id: str
    span_id: str
    parent_id: str = ""
    name: str
    start_ms: float
    end_ms: float = 0.0
    status: str = "ok"  # "error" when the call failed outright
    attributes: dict = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class Recorder:
    """Collects spans for one arm. Single-threaded by construction, not by locking.

    Each parallel session gets its own recorder and they are merged after the join, so
    no span list is ever shared between workers and a lock would only add contention.
    """

    def __init__(self, arm: str, model: str) -> None:
        self.arm = arm
        self.model = model
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, trace_id: str, parent_id: str = "", **attributes):
        """Open a span; the caller may mutate `.attributes` before it closes."""
        s = Span(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent_id,
            name=name,
            start_ms=time.time() * 1000,
            attributes={"hotset.arm": self.arm, "gen_ai.request.model": self.model, **attributes},
        )
        try:
            yield s
        except Exception:
            s.status = "error"
            raise
        finally:
            s.end_ms = time.time() * 1000
            self.spans.append(s)

    def save(self, tag: str) -> Path:
        TRACES.mkdir(exist_ok=True)
        path = TRACES / f"{tag}.jsonl"
        path.write_text("".join(json.dumps(s.model_dump()) + "\n" for s in self.spans))
        return path


def load(path: Path) -> list[Span]:
    return [Span(**json.loads(line)) for line in path.read_text().splitlines() if line]


def usage_attributes(usage) -> dict:
    """Token accounting under the conventional names, so cost is queryable downstream.

    The cache split is kept alongside the standard input count: the whole project turns
    on which fraction of the prefix was read rather than written, and `input_tokens`
    alone cannot express that.
    """
    return {
        "gen_ai.usage.input_tokens": usage.prompt_tokens,
        "gen_ai.usage.output_tokens": usage.output_tokens,
        "hotset.tokens.cached": usage.cached_tokens,
        "hotset.tokens.written": usage.write_tokens,
        "hotset.tokens.uncached": usage.uncached_tokens,
        "hotset.cache_hit_rate": usage.hit_rate(),
    }


def drift(spans: list[Span], key: str = "hotset.correct", window: int = 20) -> list[float]:
    """Rolling accuracy over turn spans, in order.

    A single number cannot show a run that starts strong and decays; this can. It is the
    span-level answer to whether token reduction degraded behaviour partway through.
    """
    flags = [bool(s.attributes.get(key)) for s in spans if s.name == "turn"]
    if len(flags) < window:
        return []
    return [sum(flags[i : i + window]) / window for i in range(len(flags) - window + 1)]
