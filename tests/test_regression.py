"""Regression guards that need no API access.

The project's whole premise is that the cached prefix is byte-stable and that
admission is priced correctly. Both are silently breakable by an innocuous edit — a
reworded instruction invalidates every deployed cache, and a sign error in the
economics shows up only as a larger bill. These pin them in CI.
"""

import hashlib

import pytest

from hotset.config import MODELS
from hotset.corpus.distractors import pad_catalog
from hotset.corpus.harvest import load
from hotset.layout.prompt import cached_prefix, dispatcher_tool
from hotset.policy.adaptive import HotSet
from hotset.policy.base import Plan
from hotset.policy.economics import asymptotic_rate, break_even
from hotset.policy.predictors import LRUK, Markov, Oracle

CATALOG = load()
HAIKU = MODELS["haiku"]


def _digest(blocks: list[dict]) -> str:
    return hashlib.blake2b("".join(b["text"] for b in blocks).encode(), digest_size=8).hexdigest()


def test_cached_prefix_bytes_are_pinned():
    """If this fails, every deployed prefix cache was just invalidated. That may be
    intended — but it must be a decision, not a side effect. Re-pin deliberately."""
    assert _digest(cached_prefix(Plan(index=CATALOG))) == "5eb827b8811767f0"


def test_dispatcher_bytes_are_pinned():
    """The tools field renders upstream of system, so its bytes gate the whole cache."""
    import json

    raw = json.dumps(dispatcher_tool(), sort_keys=True, separators=(",", ":"))
    assert hashlib.blake2b(raw.encode(), digest_size=8).hexdigest() == "7af8a68d8810a593"


def test_prefix_is_stable_across_calls():
    assert _digest(cached_prefix(Plan(index=CATALOG))) == _digest(cached_prefix(Plan(index=CATALOG)))


def test_catalog_order_does_not_leak_into_the_prefix():
    """Layer A is sorted, so an unordered catalog must not shift a single byte."""
    shuffled = list(reversed(CATALOG))
    assert _digest(cached_prefix(Plan(index=CATALOG))) == _digest(cached_prefix(Plan(index=shuffled)))


def test_threshold_falls_toward_the_price_ratio():
    rates = [break_even(HAIKU, 130, 389, horizon=t) / t for t in (50, 500, 5000)]
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] > asymptotic_rate(HAIKU)


def test_a_flat_workload_admits_nothing_even_with_the_future():
    """Pinned break-even values. Only a genuine economics change should move this."""
    catalog = pad_catalog(CATALOG, 200, seed=0)
    future = ["read_file", "git_add", "browser_click", "search_nodes"] * 20
    policy = HotSet(HAIKU, Oracle(future), horizon=len(future))
    for tool in future:
        policy.plan(catalog, [], tool)
        policy.observe(tool)
    assert policy.hot == []


@pytest.mark.parametrize("predictor", [LRUK(k=2), Markov()])
def test_predictors_never_promise_more_uses_than_the_horizon(predictor):
    """expected_uses is a count of turns, so it cannot exceed the number of turns."""
    for _ in range(30):
        predictor.advance()
        predictor.observe("hot_tool")
    assert predictor.expected_uses("hot_tool", 10) <= 10.0
