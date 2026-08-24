"""Workload composition: skew must change arrival mix without changing the questions."""

from hotset.corpus.harvest import load as load_catalog
from hotset.eval.tasks import Session, Turn
from hotset.eval.workload import concentration, session_server, trace

SERVERS = {t.name: t.server for t in load_catalog()}


def _session(name: str, tools: list[str]) -> Session:
    return Session(scenario=name, turns=[Turn(user=f"do {t}", tool=t) for t in tools])


def test_session_server_is_the_modal_server():
    s = _session("s", ["git_add", "git_commit", "read_file"])
    assert session_server(s, SERVERS) == "git"


def test_skew_concentrates_and_uniform_does_not():
    pool = [
        _session("git", ["git_add", "git_commit"]),
        _session("fs", ["read_file", "write_file"]),
        _session("mem", ["create_entities", "search_nodes"]),
    ]
    flat = concentration(trace(pool, SERVERS, skew=0.0, length=60, seed=1))[0]
    sharp = concentration(trace(pool, SERVERS, skew=3.0, length=60, seed=1))[0]
    assert sharp > flat


def test_trace_only_replays_existing_sessions():
    """Skew must not invent tasks: it changes how often each arrives, nothing else."""
    pool = [_session("a", ["git_add"]), _session("b", ["read_file"])]
    drawn = trace(pool, SERVERS, skew=2.0, length=30, seed=0)
    assert {s.scenario for s in drawn} <= {"a", "b"}
    assert len(drawn) == 30


def test_trace_is_seed_deterministic():
    pool = [_session(str(i), ["git_add"]) for i in range(5)]
    a = [s.scenario for s in trace(pool, SERVERS, skew=1.0, length=20, seed=7)]
    b = [s.scenario for s in trace(pool, SERVERS, skew=1.0, length=20, seed=7)]
    assert a == b


def test_concentration_reports_the_peak_window():
    calls = ["hot"] * 8 + ["cold"] * 42
    s = Session(scenario="s", turns=[Turn(user="u", tool=t) for t in calls])
    assert concentration([s])[1] == 42
