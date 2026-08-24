"""Predictor tests: the rate estimate has to mean what the controller thinks it means."""

from hotset.policy.predictors import LRUK


def _run(calls: list[str | None]) -> LRUK:
    """Replay a turn sequence; None is a turn with no tool call."""
    p = LRUK(k=2)
    for call in calls:
        p.advance()
        if call:
            p.observe(call)
    return p


def test_unseen_tool_predicts_nothing():
    assert _run(["a", "a"]).expected_uses("zzz", 10) == 0.0


def test_every_turn_predicts_about_one_per_turn():
    """A tool called on all four turns should predict roughly the horizon."""
    assert 8.0 <= _run(["a"] * 4).expected_uses("a", 10) <= 12.0


def test_single_sighting_is_damped():
    """One hit is weak evidence; undamped it would claim a call every turn."""
    assert _run(["a"]).expected_uses("a", 10) < 10.0


def test_idle_turns_decay_the_rate():
    """advance() on quiet turns is what stops a stale tool holding its slot."""
    hot = _run(["a", "a"]).expected_uses("a", 10)
    stale = _run(["a", "a", None, None, None, None]).expected_uses("a", 10)
    assert stale < hot


def test_frequent_beats_rare():
    p = _run(["a", "b", "a", "b", "a"])
    assert p.expected_uses("a", 10) > p.expected_uses("b", 10)


def test_ranked_is_deterministic_under_ties():
    """Two tools with identical histories must order by name, not dict insertion."""
    p = _run(["b", "a", "b", "a"])
    assert [n for n, _ in p.ranked(["b", "a"], 10)] == ["a", "b"]
