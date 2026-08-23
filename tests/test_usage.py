"""Cache accounting tests: these define the project's two headline metrics."""

import pytest

from hotset.config import MODELS
from hotset.runtime.openrouter import CacheUsage, parse_usage


def test_parse_usage_carves_slices_out_of_total():
    usage = parse_usage(
        {"usage": {"prompt_tokens": 10_000, "completion_tokens": 120,
                   "prompt_tokens_details": {"cached_tokens": 7_000, "cache_write_tokens": 500}}}
    )
    assert (usage.cached_tokens, usage.write_tokens, usage.uncached_tokens) == (7_000, 500, 2_500)
    assert usage.prompt_tokens == 10_000


def test_parse_usage_tolerates_providers_that_report_nothing():
    usage = parse_usage({"usage": {"prompt_tokens": 800}})
    assert usage.uncached_tokens == 800 and usage.hit_rate() == 0.0


def test_hit_rate_counts_cold_start_as_a_miss():
    """First turn is all writes: honest accounting scores that 0%, not 100%."""
    assert CacheUsage(uncached_tokens=0, cached_tokens=0, write_tokens=9_000,
                      output_tokens=0).hit_rate() == 0.0


def test_hit_rate_empty_prompt_does_not_divide_by_zero():
    assert CacheUsage(uncached_tokens=0, cached_tokens=0, write_tokens=0,
                      output_tokens=0).hit_rate() == 0.0


def test_cost_usd_prices_each_slice_at_its_own_rate():
    spec = MODELS["haiku"]  # 1.00 / 0.10 / 1.25 per MTok
    usage = CacheUsage(uncached_tokens=1_000_000, cached_tokens=1_000_000,
                       write_tokens=1_000_000, output_tokens=0)
    assert usage.cost_usd(spec) == pytest.approx(1.00 + 0.10 + 1.25)


def test_caching_beats_cold_at_steady_state():
    """The whole thesis in one assertion: a warm prefix is ~10x cheaper on Haiku."""
    spec = MODELS["haiku"]
    cold = CacheUsage(uncached_tokens=20_000, cached_tokens=0, write_tokens=0, output_tokens=0)
    warm = CacheUsage(uncached_tokens=0, cached_tokens=20_000, write_tokens=0, output_tokens=0)
    assert cold.cost_usd(spec) / warm.cost_usd(spec) == pytest.approx(spec.cost_ratio)


def test_our_cost_model_reconciles_with_openrouter_bill():
    """Guards against registry price drift: our arithmetic must match the vendor's."""
    spec = MODELS["qwen-flash"]
    usage = CacheUsage(uncached_tokens=19, cached_tokens=0, write_tokens=0,
                       output_tokens=1, reported_cost_usd=7e-07)
    assert usage.total_cost_usd(spec) == pytest.approx(usage.reported_cost_usd, rel=1e-6)
