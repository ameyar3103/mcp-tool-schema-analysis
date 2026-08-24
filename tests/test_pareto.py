"""Frontier logic: dominance has to survive the fact that accuracy is measured, not known."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_pareto import dominates, frontier, metrics, naive_frontier


def arm(cost, accuracy):
    return {"cost_per_turn": cost, "accuracy": accuracy}


def outcomes(pattern):
    """Turn keys are shared across arms, which is what makes the test paired."""
    return {(0, i): c for i, c in enumerate(pattern)}


def test_cheaper_and_tied_dominates():
    arms = {"a": arm(1.0, 0.5), "b": arm(2.0, 0.5)}
    paired = {"a": outcomes([1, 0] * 10), "b": outcomes([1, 0] * 10)}
    assert dominates("a", "b", arms, paired)
    assert not dominates("b", "a", arms, paired)


def test_a_significant_accuracy_loss_defeats_the_cost_advantage():
    """Cheap but reliably worse is a real trade-off, so both arms stay."""
    arms = {"cheap": arm(1.0, 0.2), "good": arm(2.0, 0.9)}
    paired = {"cheap": outcomes([0] * 40), "good": outcomes([1] * 40)}
    assert not dominates("cheap", "good", arms, paired)
    assert frontier(arms, paired) == {"cheap", "good"}


def test_a_nominal_accuracy_loss_inside_the_noise_does_not():
    """One discordant turn out of forty cannot support a ranking."""
    arms = {"cheap": arm(1.0, 0.475), "good": arm(2.0, 0.5)}
    cheap = [1, 0] * 20
    good = list(cheap)
    good[1] = 1  # a single turn of disagreement
    assert dominates("cheap", "good", arms, {"cheap": outcomes(cheap), "good": outcomes(good)})


def test_significance_shrinks_the_frontier_it_never_grows_it_for_ties():
    """The naive rule keeps an arm whose only claim is an unmeasurable accuracy edge."""
    arms = {"cheap": arm(1.0, 0.475), "good": arm(2.0, 0.5)}
    cheap = [1, 0] * 20
    good = list(cheap)
    good[1] = 1
    paired = {"cheap": outcomes(cheap), "good": outcomes(good)}
    assert naive_frontier(arms) == {"cheap", "good"}
    assert frontier(arms, paired) == {"cheap"}


def test_the_cheapest_arm_is_always_on_the_frontier():
    arms = {"a": arm(1.0, 0.1), "b": arm(2.0, 0.9), "c": arm(3.0, 0.95)}
    paired = {n: outcomes([1] * 40) for n in arms}
    assert "a" in frontier(arms, paired)


def test_equal_cost_arms_never_dominate_each_other():
    """Without a cost difference there is nothing to trade accuracy against."""
    arms = {"a": arm(1.0, 0.9), "b": arm(1.0, 0.1)}
    paired = {"a": outcomes([1] * 40), "b": outcomes([0] * 40)}
    assert frontier(arms, paired) == {"a", "b"}


def test_drift_reports_only_non_overlapping_halves():
    """A drift claim needs separated intervals; a nominal dip is not evidence."""
    from drift import halves

    from hotset.eval.significance import wilson
    from hotset.eval.spans import Span

    def span(i, correct):
        return Span(trace_id="t", span_id=str(i), name="turn", start_ms=float(i),
                    attributes={"hotset.correct": correct})

    clean = [span(i, i < 40) for i in range(80)]  # perfect, then total collapse
    (a, an), (b, bn) = halves(clean)
    assert (a, an, b, bn) == (40, 40, 0, 40)
    assert wilson(b, bn)[1] < wilson(a, an)[0]


def test_position_buckets_group_by_turn_index_not_session():
    from drift import by_position

    from hotset.eval.spans import Span

    spans = [
        Span(trace_id=f"s{s}", span_id=f"{s}-{t}", name="turn", start_ms=0.0,
             attributes={"hotset.turn": t, "hotset.correct": t < 2})
        for s in range(3)
        for t in range(4)
    ]
    assert by_position(spans) == {0: (3, 3), 1: (3, 3), 2: (0, 3), 3: (0, 3)}


def _chain():
    """Three arms whose ties chain: cheap ~ mid, mid ~ dear, but dear beats cheap."""
    cheap = set(range(50))
    mid = set(range(14, 50)) | set(range(60, 82))
    dear = set(range(20, 50)) | set(range(60, 96))
    pattern = {"cheap": cheap, "mid": mid, "dear": dear}
    arms = {"cheap": arm(1.0, 0.50), "mid": arm(2.0, 0.58), "dear": arm(3.0, 0.66)}
    paired = {n: {(0, i): i in s for i in range(100)} for n, s in pattern.items()}
    return arms, paired


def test_the_chain_is_really_a_chain():
    """Guards the fixture: the pathology only exists if these p-values hold."""
    from hotset.eval.significance import mcnemar

    _, paired = _chain()
    assert mcnemar(paired["cheap"], paired["mid"])[2] >= 0.05
    assert mcnemar(paired["mid"], paired["dear"])[2] >= 0.05
    assert mcnemar(paired["cheap"], paired["dear"])[2] < 0.05


def test_the_antichain_drops_an_arm_that_beats_the_survivor():
    from plot_pareto import antichain

    arms, paired = _chain()
    assert antichain(arms, paired) == {"cheap"}


def test_the_frontier_puts_that_arm_back():
    """A recommendation that omits a measurably better option is not a recommendation."""
    from plot_pareto import frontier

    arms, paired = _chain()
    assert frontier(arms, paired) == {"cheap", "dear"}


def test_re_admission_reaches_a_fixpoint():
    from plot_pareto import frontier, inversions

    arms, paired = _chain()
    best = frontier(arms, paired)
    assert inversions(arms, paired, best) == []


def test_relative_drift_cancels_a_shift_shared_by_every_arm():
    """Both arms improve in the second half; that is the suite, not either agent."""
    from drift import relative

    arm_a = {(0, i): i >= 50 for i in range(100)}
    reference = {(0, i): i >= 50 for i in range(100)}
    _, _, p = relative(arm_a, reference)
    assert p == 1.0


def test_relative_drift_catches_a_shift_in_only_one_arm():
    """The reference holds steady while the arm collapses after the midpoint."""
    from drift import relative

    reference = dict.fromkeys(((0, i) for i in range(100)), True)
    arm_a = {(0, i): i < 50 for i in range(100)}
    first, second, p = relative(arm_a, reference)
    assert first == (0, 50) and second == (50, 50)
    assert p < 0.001


def _result(session, turn, correct, twin, cost):
    """A minimal Result-alike; metrics() only reads these fields."""
    return SimpleNamespace(
        session=session, turn=turn, correct=correct, twin=twin, error=None,
        cost_usd=cost, hit_rate=lambda: 0.9,
    )


def test_lenient_scoring_moves_cost_per_correct_with_the_accuracy():
    """A row must not pair a lenient interval with a strict $/correct."""
    rows = [_result(0, i, i < 2, 2 <= i < 4, 0.001) for i in range(10)]
    strict = metrics(rows)
    assert strict["strict"] == pytest.approx(0.2)
    assert strict["lenient"] == pytest.approx(0.4)
    # main() rewrites the lenient row; $/correct has to halve, not stay at the strict rate
    assert strict["cost_per_correct"] == pytest.approx(strict["cost_per_turn"] / 0.2)
    assert strict["cost_per_turn"] / strict["lenient"] == pytest.approx(
        strict["cost_per_turn"] / 0.4
    )


def test_strict_survives_the_lenient_rewrite():
    """`accuracy` is overwritten in lenient mode; `strict` is what the report reads."""
    rows = [_result(0, i, i < 3, 3 <= i < 5, 0.001) for i in range(10)]
    m = metrics(rows)
    rewritten = dict(m, accuracy=m["lenient"], lo=m["lenient_lo"], hi=m["lenient_hi"])
    assert rewritten["accuracy"] == pytest.approx(0.5)
    assert rewritten["strict"] == pytest.approx(0.3)
