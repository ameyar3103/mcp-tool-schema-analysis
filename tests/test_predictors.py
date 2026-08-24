"""Predictor tests: the rate estimate has to mean what the controller thinks it means."""

import pytest

from hotset.policy.predictors import LRUK, Ensemble, Markov, Oracle, warm


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


def test_markov_prefers_the_observed_successor():
    m = Markov()
    warm(m, [["read_file", "edit_file"]] * 10)
    m.reset()
    m.advance()
    m.observe("read_file")
    assert m.expected_uses("edit_file", 5) > m.expected_uses("read_file", 5)


def test_markov_falls_back_to_the_marginal_without_context():
    m = Markov()
    warm(m, [["a", "b"], ["a", "c"]])
    m.reset()
    assert m.expected_uses("a", 10) == pytest.approx(10 * m._marginal("a"))


def test_markov_does_not_chain_across_sessions():
    """Last tool of one scenario must not become a predictor of the next scenario's first."""
    chained, split_ = Markov(), Markov()
    warm(chained, [["a", "b", "z", "y"]])
    warm(split_, [["a", "b"], ["z", "y"]])
    assert chained.edges["b"]["z"] == 1
    assert "z" not in split_.edges.get("b", {})


def test_markov_horizon_is_bounded_by_the_horizon():
    """A probability sum over H turns cannot exceed H uses."""
    m = Markov()
    warm(m, [["a", "a", "a", "a"]] * 5)
    m.reset()
    m.advance()
    m.observe("a")
    assert 0.0 <= m.expected_uses("a", 20) <= 20.0


def test_ensemble_averages_member_estimates():
    lru, mk = LRUK(), Markov()
    ens = Ensemble([lru, mk])
    warm(ens, [["a", "b"], ["a", "b"]])
    expected = (lru.expected_uses("a", 10) + mk.expected_uses("a", 10)) / 2
    assert ens.expected_uses("a", 10) == pytest.approx(expected)


def test_oracle_counts_the_actual_future():
    o = Oracle(["a", "b", "a", "c", "a"])
    o.advance()  # planning turn 0
    assert o.expected_uses("a", 5) == 3.0
    o.advance()  # planning turn 1
    assert o.expected_uses("a", 4) == 2.0


def test_oracle_sees_the_turn_being_planned():
    """The tool about to be called is exactly the one admission should catch."""
    o = Oracle(["target"])
    o.advance()
    assert o.expected_uses("target", 50) == 1.0


def test_oracle_dominates_a_learned_predictor_on_a_novel_tool():
    """A tool never seen before scores zero under LRU-K and its true rate under oracle."""
    future = ["new_tool"] * 4
    o = Oracle(future)
    o.advance()
    assert o.expected_uses("new_tool", 4) == 4.0
    assert LRUK().expected_uses("new_tool", 4) == 0.0
