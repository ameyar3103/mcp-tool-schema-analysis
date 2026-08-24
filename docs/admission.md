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

**Full catalog collapses at scale.** 86.0% at 76 tools → 45.3% at 300, while
remaining the most expensive arm — the documented context-rot failure, reproduced.
Sending everything is not a strong baseline, it is the worst one on cost per
correct call by a factor of 4.7.

**Placement beats retrieval.** RAG-over-tools and HotSet run the *same* BM25 over
the *same* catalog. RAG rebuilds the prefix around the results (11.8% hit, 36.8%
accurate); HotSet appends them as a suffix behind a stable prefix (94.4% hit, 68.4%
accurate). Nothing about the retrieval changed — only where the bytes went.

**The tail is free.** `hotset` and `index only` cost exactly the same $0.000093,
and differ only by three tail-loaded schemas per turn. Those three schemas are worth
9.5 accuracy points at no measurable cost, because a suffix does not disturb the
cached prefix.

**The controller admitted nothing, correctly.** Alibaba ignores the second
breakpoint, so `S` is the entire 8,570-token prefix and `n*` ≈ 74 — above the 50
uses a 50-turn horizon can possibly supply. HotSet therefore reduces to index + tail
on this provider. That is the economics working, not failing: on a model with cheap
uncached tokens and no breakpoint control, caching schemas genuinely does not pay.

**Honest Pareto.** Lazy discovery is more accurate (76.8%) at 38% higher cost and
1.05 extra round trips per turn. It is not dominated, and should not be reported as
if it were. HotSet dominates every other arm; against lazy discovery it trades 8.4
accuracy points for 27% lower cost per correct call.
