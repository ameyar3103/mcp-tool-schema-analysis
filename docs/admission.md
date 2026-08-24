# Week 3 — Admission economics

## The break-even, derived rather than assumed

Over a horizon of `T` turns, for a tool whose schema is `P` tokens sitting in a
cache segment of `S` tokens, with per-token prices `u` (uncached), `c` (cached) and
`w` (write):

```
admitted      = S·w + (T-1)·S·c
not admitted  = T·(S-P)·c + n·P·u
```

Setting `admitted ≤ not admitted` and solving for the use count `n`:

```
n* = S·(w - c) / (P·u)  +  T·c/u
```

This reproduces both admission costs measured against live Anthropic endpoints in
Q7 — 5.75 uses with split breakpoints (`S`=767) and 54.8 without (`S`=7420) — and
those two cases are pinned as tests.

Two consequences the original sketch missed:

- **`P` is in the denominator, so the threshold is per tool.** A large schema is
  expensive to tail-load repeatedly and therefore earns admission *sooner* than a
  small one. A frequency-ranked policy cannot see this.
- **`S` is the rewritten segment, not the prefix.** Where a provider honours a
  second breakpoint that is layer B alone; where it does not it is the whole prefix.
  Same tool, same traffic, 9.5× different threshold.

## Token accounting

Counts now come from a real BPE rather than `chars/3.7`, which ran ~25% high:

| | chars/3.7 | measured | |
|---|---|---|---|
| index line | 39 | **28.2** | layer A, per tool |
| full schema | 156 | **124.5** | layer B, per tool |
| compression | ~4.0× | **4.42×** | |

Anthropic publishes no tokenizer, so each `ModelSpec` carries a scale factor fitted
against live usage (R² ≈ 1.00 over three prefix sizes):

| model | scale vs reference | tools-field overhead |
|---|---|---|
| qwen3.7-flash | 1.0226 | **304 tok** |
| claude-haiku-4.5 | 1.1502 | **591 tok** |

The intercept is provider-injected tool-use boilerplate — roughly five schemas'
worth of fixed cost incurred by having a `tools` field at all, before any tool.

## Sweep — qwen3.7-flash, 300 tools, 95 held-out turns

| arm | accuracy | halluc | hit | prompt tok | hops | $/turn | **$/correct** |
|---|---|---|---|---|---|---|---|
| lazy discovery | **76.8%** | 0.0% | 16.1% | 4,477 | 1.05 | $0.000128 | $0.000167 |
| **hotset (LRU-K)** | 68.4% | 0.0% | 94.4% | 11,819 | 0.00 | $0.000093 | **$0.000135** |
| static hot set | 66.3% | 0.0% | 95.5% | 13,279 | 0.00 | $0.000103 | $0.000155 |
| index only | 58.9% | 2.1% | 94.1% | 11,344 | 0.00 | $0.000093 | $0.000159 |
| full catalog | 45.3% | 7.4% | 95.3% | 37,537 | 0.00 | $0.000286 | $0.000631 |
| RAG-over-tools | 36.8% | 18.9% | 11.8% | 1,875 | 0.00 | $0.000070 | $0.000190 |

**Full catalog collapses at scale.** 86.0% at 76 tools → 45.3% at 300 (unmatched
held-out sets, so read the 41-point drop as an effect and not a measurement), while
remaining the most expensive arm — the documented context-rot failure, reproduced.
Sending everything is not a strong baseline, it is the worst one on cost per
correct call by a factor of 4.7.

**Placement beats retrieval.** RAG-over-tools and HotSet run the *same* BM25 over
the *same* catalog. RAG rebuilds the prefix around the results (11.8% hit, 36.8%
accurate); HotSet appends them as a suffix behind a stable prefix (94.4% hit, 68.4%
accurate). Nothing about the retrieval changed — only where the bytes went.

**The tail is free.** `hotset` and `index only` cost exactly the same $0.000093,
and differ only by three tail-loaded schemas per turn. The 9.5-point accuracy gap is
*not* significant (McNemar 18/9, p=0.122), so the honest claim is the cost one: a
suffix buys schemas without disturbing the cached prefix, and the accuracy effect is
directionally positive but under-powered at n=95.

**The controller admitted nothing, correctly.** Alibaba ignores the second
breakpoint, so `S` is the entire 8,570-token prefix and `n*` ≈ 74 — above the 50
uses a 50-turn horizon can possibly supply. HotSet therefore reduces to index + tail
on this provider. That is the economics working, not failing: on a model with cheap
uncached tokens and no breakpoint control, caching schemas genuinely does not pay.

**Honest Pareto.** Lazy discovery is more accurate (76.8%) at 38% higher cost and
1.05 extra round trips per turn. It is not dominated, and should not be reported as
if it were. HotSet dominates every other arm; against lazy discovery it trades 8.4
accuracy points for 27% lower cost per correct call.

## Sweep — claude-haiku-4.5, 300 tools, 95 held-out turns

| arm | accuracy | halluc | hit | prompt tok | hops | $/turn | **$/correct** |
|---|---|---|---|---|---|---|---|
| index only | **81.1%** | 0.0% | 92.3% | 13,116 | 0.00 | $0.002785 | **$0.003436** |
| **hotset (LRU-K)** | 75.8% | 0.0% | 90.8% | 14,049 | 0.00 | $0.003103 | $0.004095 |
| static hot set | 75.8% | 0.0% | 95.6% | 15,478 | 0.00 | $0.002649 | $0.003495 |
| full catalog | 73.7% | 0.0% | 94.5% | 43,785 | 0.00 | $0.007385 | $0.010023 |
| lazy discovery | 73.7% | 0.0% | **0.0%** | 5,299 | 0.92 | $0.006022 | $0.008173 |
| RAG-over-tools | 38.9% | 0.0% | **0.0%** | 2,280 | 0.00 | $0.002841 | $0.007295 |

**Admission fires here, and the closed form said it would.** Same catalog, same
traffic, same predictor — HotSet admitted nothing on Alibaba and three tools on
Anthropic. The only thing that changed is who honours a second cache breakpoint:

| tool | schema `P` | `n*` on Haiku | `n*` on qwen-flash |
|---|---|---|---|
| `enumerate_directory_with_sizes` | 159 | 7.8 | 98.1 |
| `move_file` | 124 | 8.6 | 122.9 |
| `get_file_info` | 106 | 9.2 | 142.1 |

With four breakpoints the rewritten segment `S` is the 389-token hot block; with one
it is the whole 11,641-token prefix. A 50-turn horizon can supply at most 50 uses, so
one provider clears the bar by 6× and the other misses it by 2×. No threshold was
tuned per model: `n* = S·(w−c)/(P·u) + T·c/u` was derived in week 1 and both
decisions fall out of it.

**Token reduction has a floor, and past it caching stops existing.** RAG-over-tools
and lazy discovery post **0.0%** hit rates — not a bug, and not the 11–16% they score
on Alibaba. Haiku's minimum cacheable prefix is 4,096 tokens; both arms build
prefixes below it, so caching silently no-ops and every token is billed at the
uncached rate. RAG ends up *more expensive per correct call* than three arms that
send 6× more tokens. Shrinking the prompt is only a win above the provider's floor.

**At n=95, the top five arms are one arm.** Paired McNemar (`scripts/significance.py`)
separates only RAG-over-tools; every other pair sits at p ≥ 0.118, and the smallest
gap this suite can detect at 80% power is 7.2%. The 81.1% vs 73.7% spread is real
enough to act on and too small to publish as a ranking. What *is* separable is cost:
$0.003436 to $0.010023 per correct call, a 2.9× spread at indistinguishable accuracy.

**Full catalog is the arm caching cannot save.** It hits 94.5% and still costs 2.9×
the leader, because 94.5% of 43,785 tokens is more absolute spend than 92.3% of
13,116. Hit rate is a ratio; the bill is not.

**Hallucination looks like a capability, not a layout property.** Every Haiku arm
scored 0.0%, including full catalog and RAG, which hallucinated 7.4% and 18.9% on
qwen-flash. Stated cautiously because the two sweeps did not share a held-out set
(see below); the effect is far larger than the split could explain, but it is not
yet a matched comparison.

> Both sweeps ran on distractor pool **v1**, which sampled differentiator clauses
> uniformly and so occasionally handed a browser tool "Fails instead of overwriting
> existing entries". v2 keys the pool by server. Padded catalogs are built at run time,
> so v1 and v2 numbers must not be pooled — `CORPUS_VERSION` records which is live.

> **Split defect, and what it does and does not invalidate.** `split()` keyed on
> `hash((seed, scenario))`, and CPython salts string hashing per process, so these two
> sweeps drew different held-out sets — they share only 6 of 95 ground-truth labels.
> Within a sweep all six arms run in one process and one split, so every paired McNemar
> result above stands. What does not stand is any comparison of a qwen-flash accuracy to
> a haiku accuracy. The break-even table is unaffected: `n*` depends on token counts,
> prices and the horizon, not on which turns were sampled. Fixed with a blake2b key and
> a subprocess regression test; both sweeps are being re-run on the stable split.
