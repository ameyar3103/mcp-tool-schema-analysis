"""Span instrumentation: the trace has to survive JSON and preserve ordering."""

import json

from hotset.eval.spans import Recorder, Span, drift
from hotset.runtime.openrouter import CacheUsage


def test_spans_nest_under_their_parent():
    rec = Recorder("hotset", "qwen")
    with rec.span("turn", "t1") as turn, rec.span("chat", "t1", turn.span_id):
        pass
    child, parent = rec.spans  # children close first
    assert child.parent_id == parent.span_id
    assert child.trace_id == parent.trace_id == "t1"


def test_a_failed_span_is_marked_and_still_recorded():
    """A transport failure must appear in the trace, not vanish from it."""
    rec = Recorder("hotset", "qwen")
    try:
        with rec.span("chat", "t1"):
            raise TimeoutError
    except TimeoutError:
        pass
    assert rec.spans[0].status == "error"
    assert rec.spans[0].end_ms > 0


def test_arm_and_model_are_stamped_on_every_span():
    rec = Recorder("rag-over-tools", "haiku")
    with rec.span("turn", "t1"):
        pass
    assert rec.spans[0].attributes["hotset.arm"] == "rag-over-tools"
    assert rec.spans[0].attributes["gen_ai.request.model"] == "haiku"


def test_span_round_trips_through_json():
    rec = Recorder("hotset", "qwen")
    with rec.span("turn", "t1", **{"hotset.correct": True}):
        pass
    assert Span(**json.loads(json.dumps(rec.spans[0].model_dump()))) == rec.spans[0]


def test_usage_attributes_keep_the_cache_split():
    from hotset.eval.spans import usage_attributes

    u = CacheUsage(uncached_tokens=100, cached_tokens=900, write_tokens=0, output_tokens=10)
    attrs = usage_attributes(u)
    assert attrs["gen_ai.usage.input_tokens"] == 1000
    assert attrs["hotset.cache_hit_rate"] == 0.9


def test_drift_exposes_decay_a_mean_would_hide():
    """Half right then half wrong averages to 50%; the rolling view shows the cliff."""
    spans = [
        Span(trace_id="t", span_id=str(i), name="turn", start_ms=0, attributes={"hotset.correct": i < 40})
        for i in range(80)
    ]
    series = drift(spans, window=20)
    assert series[0] == 1.0
    assert series[-1] == 0.0


def test_drift_is_empty_below_the_window():
    spans = [Span(trace_id="t", span_id="1", name="turn", start_ms=0, attributes={})]
    assert drift(spans, window=20) == []
