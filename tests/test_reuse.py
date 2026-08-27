"""Tool-level reuse: the workload knob that server skew structurally cannot supply."""

from collections import Counter

from hotset.eval.tasks import Session, Turn
from hotset.eval.workload import concentration, repeat


def session(scenario, pairs):
    return Session(scenario=scenario, turns=[Turn(user=u, tool=t) for u, t in pairs])


def suite():
    """Two sessions per tool, so every tool has an alternate phrasing to draw on."""
    return [
        session("a", [("open the file", "read"), ("save it", "write")]),
        session("b", [("show me the contents", "read"), ("persist that", "write")]),
    ]


def test_rate_zero_returns_the_suite_untouched():
    s = suite()
    assert repeat(s, rate=0.0) is s


def test_reuse_only_repeats_tools_already_in_that_session():
    """A repeat must deepen the session's working set, never widen it."""
    s = [session("a", [("x", "read"), ("y", "read")]), session("b", [("z", "write")])]
    for got, before in zip(repeat(s, rate=2.0, seed=1), s):
        assert {t.tool for t in got.turns} == {t.tool for t in before.turns}


def test_repeats_borrow_another_sessions_wording():
    """Duplicating the same user text would make the repeat trivially easy."""
    out = repeat(suite(), rate=2.0, seed=0)
    reads = [t.user for s in out for t in s.turns if t.tool == "read"]
    assert len(set(reads)) > 1


def test_a_tool_with_no_alternate_phrasing_is_never_repeated():
    """One phrasing means any repeat would be a verbatim duplicate, so skip it."""
    s = [session("a", [("only wording", "solo")])]
    assert [t.user for t in repeat(s, rate=3.0, seed=0)[0].turns] == ["only wording"]


def test_the_cap_bounds_how_hot_one_tool_gets():
    """Uncapped, a long session collapses onto a single tool and stops being a workload."""
    s = [session("a", [(f"q{i}", "read") for i in range(4)]),
         session("b", [("other wording", "read")])]
    counts = Counter(t.tool for t in repeat(s, rate=5.0, seed=0, cap=4)[0].turns)
    assert counts["read"] == 4


def spread(n_tools=30):
    """Each tool used by exactly two sessions: diverse, but every tool has an alternate."""
    return [
        session(f"s{i}{half}", [(f"q{i}{half}", f"tool{i}")])
        for i in range(n_tools)
        for half in "ab"
    ]


def test_reuse_lifts_peak_uses_per_horizon():
    """The whole point: raise per-tool uses per horizon, which server skew cannot."""
    base = spread()
    assert concentration(base)[1] == 2  # every tool appears exactly twice
    assert concentration(repeat(base, rate=3.0, seed=0))[1] > 2


def test_the_frozen_suite_goes_from_under_the_break_even_to_over_it():
    """The claim the sweep rests on, checked against the suite the sweep actually runs."""
    from hotset.eval.tasks import load, split

    _, test = split(load())
    assert concentration(test)[1] < 8  # n* on Haiku is 7.8-9.2 over a 50-turn horizon
    assert concentration(repeat(test, rate=1.0, seed=0))[1] >= 8


def test_reuse_is_deterministic_under_a_seed():
    a = repeat(suite(), rate=2.0, seed=7)
    b = repeat(suite(), rate=2.0, seed=7)
    assert [[t.user for t in s.turns] for s in a] == [[t.user for t in s.turns] for s in b]
