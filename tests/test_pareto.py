"""Frontier logic: dominance has to survive the fact that accuracy is measured, not known."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_pareto import dominates, frontier, naive_frontier


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
