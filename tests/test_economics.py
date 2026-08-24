"""Break-even tests, pinned to costs measured against live providers."""


import math

import pytest

from hotset.config import MODELS
from hotset.policy.economics import (
    admission_cost,
    amortization_horizon,
    asymptotic_rate,
    break_even,
    rewritten_segment,
    tail_cost,
)

HAIKU = MODELS["haiku"]
SCHEMA = 156  # tokens in one tail-loaded schema, as measured in the Q7 run


def test_break_even_matches_measured_split_admission():
    """Haiku honours the second breakpoint: admission rewrites layer B alone (767 tok)."""
    assert round(break_even(HAIKU, SCHEMA, 767), 2) == 5.75


def test_break_even_matches_measured_single_breakpoint():
    """Without the split, admission rewrites the whole 7420-token prefix."""
    assert round(break_even(HAIKU, SCHEMA, 7420), 1) == 54.8


def test_split_is_worth_roughly_ten_times():
    """The headline of Q7: a purely structural choice moves the threshold 9.5x."""
    ratio = break_even(HAIKU, SCHEMA, 7420) / break_even(HAIKU, SCHEMA, 767)
    assert 9.0 < ratio < 10.0


def test_horizon_raises_the_threshold():
    """An admitted schema is re-read every turn, so a long horizon makes admission dearer."""
    assert break_even(HAIKU, SCHEMA, 767, horizon=20) > break_even(HAIKU, SCHEMA, 767, horizon=1)


def test_segment_depends_on_provider_breakpoint_support():
    """Alibaba ignores the second breakpoint, so the whole prefix is at risk."""
    assert rewritten_segment(MODELS["haiku"], 6652, 767) == 767
    assert rewritten_segment(MODELS["qwen-flash"], 6652, 767) == 7419


def test_cheap_model_admits_sooner_than_expensive_one():
    """Qwen's 5x uncached/cached ratio vs Haiku's 10x is a real experimental variable."""
    qwen, haiku = MODELS["qwen-flash"], MODELS["haiku"]
    assert break_even(qwen, SCHEMA, 767) != break_even(haiku, SCHEMA, 767)


def test_costs_are_positive_and_ordered():
    """A tail-load must be cheaper than a full segment rewrite, or nothing else follows."""
    assert 0 < tail_cost(HAIKU, SCHEMA) < admission_cost(HAIKU, 767)


def test_zero_schema_never_admits():
    """Guard against a divide-by-zero turning into a free admission."""
    assert math.isinf(break_even(HAIKU, 0, 767))


def test_asymptotic_rate_is_the_price_ratio():
    assert asymptotic_rate(HAIKU) == pytest.approx(HAIKU.cached / HAIKU.uncached)


def test_break_even_rate_converges_to_the_asymptote():
    """n*/T must fall toward c/u as the horizon grows, from above."""
    rates = [break_even(HAIKU, 130, 389, horizon=t) / t for t in (50, 200, 1000, 5000)]
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] == pytest.approx(asymptotic_rate(HAIKU), abs=2e-3)


def test_no_horizon_rescues_a_rate_below_the_bound():
    below = asymptotic_rate(HAIKU) * 0.9
    assert amortization_horizon(HAIKU, 130, 389, below) == math.inf


def test_a_bigger_rewrite_takes_longer_to_amortize():
    fast = amortization_horizon(HAIKU, 130, 389, 0.3)
    slow = amortization_horizon(HAIKU, 130, 8000, 0.3)
    assert math.isfinite(fast) and slow > fast
