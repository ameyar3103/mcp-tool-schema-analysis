import pytest

from hotset.eval.runner import TurnResult
from hotset.eval.significance import compare, fisher, mcnemar, minimum_detectable, outcomes, wilson


def _res(session: int, turn: int, correct: bool, error: str = "") -> TurnResult:
    return TurnResult(
        arm="a", model="m", session=session, turn=turn, expected="x",
        predicted="x" if correct else "y", correct=correct, hallucinated=False,
        hops=0, cached=0, written=0, uncached=0, output=0,
        cost_usd=0.0, latency_s=0.0, error=error,
    )


def test_errored_turns_are_dropped_not_scored_wrong():
    got = outcomes([_res(1, 0, False, error="429"), _res(1, 1, True)])
    assert got == {(1, 1): True}


def test_identical_arms_are_never_significant():
    a = {(1, i): i % 2 == 0 for i in range(20)}
    assert mcnemar(a, dict(a)) == (0, 0, 1.0)


def test_mcnemar_ignores_agreed_turns():
    """Adding turns both arms get right must not change the p-value."""
    a = {(1, 0): True, (1, 1): True}
    b = {(1, 0): False, (1, 1): True}
    base = mcnemar(a, b)[2]
    a |= {(1, i): True for i in range(2, 200)}
    b |= {(1, i): True for i in range(2, 200)}
    assert mcnemar(a, b)[2] == base


def test_lopsided_discordance_is_significant():
    a = {(1, i): True for i in range(10)}
    b = {(1, i): False for i in range(10)}
    _, _, p = mcnemar(a, b)
    assert p < 0.01


def test_wilson_stays_inside_the_unit_interval_at_the_ceiling():
    lo, hi = wilson(20, 20)
    assert 0.0 < lo < 1.0 and hi <= 1.0


def test_compare_orders_the_stronger_arm_first():
    strong = [_res(1, i, True) for i in range(10)]
    weak = [_res(1, i, i < 3) for i in range(10)]
    (c,) = compare({"weak": weak, "strong": strong})
    assert (c.a, c.b) == ("strong", "weak")
    assert c.a_only == 7 and c.b_only == 0


def test_power_improves_with_sample_size():
    assert minimum_detectable(400) < minimum_detectable(95)


def test_split_is_stable_across_processes():
    """Regression: `hash()` is salted per process, so two sweeps scored different turns."""
    import subprocess
    import sys

    code = "from hotset.eval.tasks import load, split; print([s.scenario for s in split(load())[1]])"
    runs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        for _ in range(3)
    }
    assert len(runs) == 1


def test_twin_is_scored_apart_from_correct_and_hallucinated():
    """A synthetic near-duplicate is neither right nor an invented name."""
    from hotset.eval.runner import summarize

    r = _res(1, 0, False)
    r.predicted, r.twin = "aux_git_add", True
    m = summarize([r])
    assert m["accuracy"] == 0.0
    assert m["twin"] == 1.0
    assert m["lenient_accuracy"] == 1.0
    assert m["hallucinated"] == 0.0


def test_fisher_matches_the_tea_tasting_table():
    """The textbook 2x2; a wrong tail sum shows up here immediately."""
    assert fisher(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)


def test_fisher_is_symmetric_under_row_and_column_swaps():
    assert fisher(3, 1, 1, 3) == pytest.approx(fisher(1, 3, 3, 1))
    assert fisher(7, 2, 3, 8) == pytest.approx(fisher(3, 8, 7, 2))


def test_fisher_separates_a_clean_split():
    assert fisher(10, 0, 0, 10) < 0.001


def test_fisher_is_one_when_the_proportions_match():
    assert fisher(5, 5, 5, 5) == pytest.approx(1.0)


def test_fisher_handles_an_empty_margin():
    assert fisher(0, 0, 4, 4) == 1.0
