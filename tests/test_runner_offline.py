"""The session loop, run without a network: trace shape and twin scoring.

Everything else about the runner is exercised only by paid sweeps, which means a defect
in the instrumentation shows up hours and dollars later. A stubbed transport puts the
whole loop — spans, hops, twin provenance — inside the offline suite.
"""

import json

import pytest

from hotset.config import MODELS
from hotset.corpus.models import Tool
from hotset.eval import runner
from hotset.eval.tasks import Session, Turn
from hotset.policy.baselines import IndexOnly
from hotset.runtime.openrouter import CacheUsage

CATALOG = [
    Tool(name="file_read", description="read a file", server="filesystem"),
    Tool(name="file_read_alt", description="read a file", server="filesystem-alt",
         synthetic=True, twin_of="file_read"),
    Tool(name="git_commit", description="commit", server="git"),
]
SESSIONS = [
    Session(scenario="a", turns=[Turn(user="read it", tool="file_read"),
                                 Turn(user="commit it", tool="git_commit")]),
    Session(scenario="b", turns=[Turn(user="read again", tool="file_read")]),
]


def stub(answer):
    """Replaces the transport with a fixed tool call, so no key or network is needed."""
    usage = CacheUsage(uncached_tokens=100, cached_tokens=900, write_tokens=0, output_tokens=5)
    message = {"tool_calls": [{"function": {"name": answer, "arguments": "{}"}}]}
    return lambda spec, frag, sid: (message, usage, 0.01)


@pytest.fixture
def recorder():
    return runner.Recorder("index-only", "qwen-flash")


def run(monkeypatch, answer, recorder):
    monkeypatch.setattr(runner, "_call", stub(answer))
    return runner.run_arm(IndexOnly(), MODELS["qwen-flash"], CATALOG, SESSIONS,
                          workers=2, recorder=recorder)


def test_every_turn_produces_one_turn_span_and_one_chat_span(monkeypatch, recorder):
    run(monkeypatch, "file_read", recorder)
    turns = [s for s in recorder.spans if s.name == "turn"]
    chats = [s for s in recorder.spans if s.name == "chat"]
    assert len(turns) == len(chats) == 3  # three labeled turns across two sessions


def test_chat_spans_nest_under_the_turn_that_issued_them(monkeypatch, recorder):
    run(monkeypatch, "file_read", recorder)
    ids = {s.span_id for s in recorder.spans if s.name == "turn"}
    assert all(s.parent_id in ids for s in recorder.spans if s.name == "chat")


def test_each_session_is_its_own_trace(monkeypatch, recorder):
    """Sessions must not share a trace id, or drift analysis crosses conversations."""
    run(monkeypatch, "file_read", recorder)
    assert len({s.trace_id for s in recorder.spans}) == len(SESSIONS)


def test_usage_lands_on_the_chat_span(monkeypatch, recorder):
    run(monkeypatch, "file_read", recorder)
    chat = next(s for s in recorder.spans if s.name == "chat")
    assert chat.attributes["gen_ai.usage.input_tokens"] == 1000
    assert chat.attributes["hotset.cache_hit_rate"] == pytest.approx(0.9)


def test_a_twin_answer_is_scored_as_twin_not_as_wrong(monkeypatch, recorder):
    """The distractor's description is identical; only the label separates them."""
    results = run(monkeypatch, "file_read_alt", recorder)
    reads = [r for r in results if r.expected == "file_read"]
    assert all(r.twin and not r.correct for r in reads)


def test_an_unrelated_wrong_answer_is_not_a_twin(monkeypatch, recorder):
    results = run(monkeypatch, "git_commit", recorder)
    reads = [r for r in results if r.expected == "file_read"]
    assert all(not r.twin and not r.correct for r in reads)


def test_spans_survive_the_save_load_round_trip(monkeypatch, recorder, tmp_path):
    run(monkeypatch, "file_read", recorder)
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps(s.model_dump()) + "\n" for s in recorder.spans))
    from hotset.eval.spans import load

    assert load(path) == recorder.spans


def test_run_arm_without_a_recorder_still_returns_results(monkeypatch):
    """Span collection is opt-in; the paid sweeps that predate it must keep working."""
    monkeypatch.setattr(runner, "_call", stub("file_read"))
    assert len(runner.run_arm(IndexOnly(), MODELS["qwen-flash"], CATALOG, SESSIONS)) == 3
