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

## Sweep — qwen3.7-flash, 300 tools, 310 held-out turns (salt `97b45e2a`)

| arm | strict | lenient | hit | prompt tok | hops | $/turn | $/correct |
|---|---|---|---|---|---|---|---|
| lazy discovery | **55.8%** | **61.9%** | 15.7% | 4,262 | 0.96 | $0.000123 | $0.000220 |
| static hot set | 43.5% | 58.1% | 97.4% | 13,769 | 0.00 | $0.000099 | **$0.000227** |
| full catalog | 36.5% | 57.7% | 98.5% | 39,193 | 0.00 | $0.000258 | $0.000709 |
| RAG-over-tools | 33.2% | 52.3% | 5.1% | 2,060 | 0.00 | **$0.000080** | $0.000242 |
| index only | 31.6% | 54.5% | 97.2% | 11,793 | 0.00 | $0.000086 | $0.000271 |
| hotset (LRU-K) | 31.0% | **61.9%** | 94.0% | 12,357 | 0.00 | $0.000099 | $0.000321 |

**The controller admitted nothing, correctly.** Alibaba ignores the second cache
breakpoint, so `S` is the whole 11,641-token prefix and `n*` lands at 98–142 uses over a
50-turn horizon. The workload's peak use of any tool in any 50-turn window is 4. The
oracle predictor, given the actual future, also admits nothing — which is how we know
this is the economics refusing rather than the predictor failing.

**Placement beats retrieval.** RAG-over-tools and HotSet run the *same* BM25 index over
the *same* catalog. RAG rebuilds the prefix around its results (5.1% hit); HotSet appends
a suffix behind a stable prefix (94.0% hit). Nothing about the retrieval changed — only
where the bytes went. On lenient accuracy HotSet leads by 9.6 points (p=0.0032).

**Sending everything is not a strong baseline.** The full catalog is the most expensive
arm by 2.1–3.2× and is not more accurate than HotSet (57.7% vs 61.9% lenient, p=0.223)
while sending 3.2× the tokens. Its 98.5% hit rate does not save it: a hit rate is a
ratio, and the bill is not.

**On lenient accuracy the frontier is `RAG-over-tools`, `hotset`, `lazy discovery`.**
`index-only` and `static hot set` fall off — cheaper arms tie them.

That frontier is not evidence for admission, and it should not be read as such. The hot
set is empty on this provider, so the `hotset` arm here *is* layer A plus a BM25 suffix
with the controller inert. What it demonstrates is the placement effect above, nothing
more. Its strict accuracy is 31.0%, second-worst in the table, 12.6 points behind
`static hot set` (p=0.000) — an arm that ships a fixed hot block and never adapts.

## Sweep — claude-haiku-4.5, 300 tools, 310 held-out turns (salt `52cbabb1`)

| arm | strict | twin | lenient | hit | prompt tok | hops | $/turn | $/correct |
|---|---|---|---|---|---|---|---|---|
| static hot set | **42.9%** | 11.9% | 54.8% | 96.4% | 15,954 | 0.00 | $0.002662 | **$0.006204** |
| full catalog | 41.6% | 14.8% | **56.5%** | 97.9% | 45,681 | 0.00 | $0.006011 | $0.014444 |
| index only | 37.7% | 18.7% | **56.5%** | 94.7% | 13,651 | 0.00 | **$0.002599** | $0.006888 |
| lazy discovery | 36.8% | 7.7% | 44.5% | 0.0% | 5,130 | 0.85 | $0.005976 | $0.016251 |
| hotset (LRU-K) | 31.6% | 21.9% | 53.5% | 91.4% | 14,572 | 0.00 | $0.003184 | $0.010071 |
| RAG-over-tools | 22.6% | 8.4% | 31.0% | 0.9% | 2,531 | 0.00 | $0.003219 | $0.014254 |

**Layer B buys nothing on this model.** `index-only` and `full-catalog` both score
**175/310** lenient — 16 discordant pairs each way, **p = 1.0000**. Dropping every JSON
Schema and keeping only tool names with argument names costs exactly zero accuracy, at
**43% of the cost and 30% of the prompt tokens**. That is an exact tie, not rounding.

**Admission fires here, as the closed form said it would.** Same catalog, same traffic,
same predictor: HotSet admitted nothing on Alibaba and three tools on Anthropic
(`add_observations`, `create_entities`, `find_nodes`). The only thing that changed is
who honours a second cache breakpoint.

| tool | schema `P` | `n*` on Haiku | `n*` on qwen-flash |
|---|---|---|---|
| `enumerate_directory_with_sizes` | 159 | 7.8 | 98.1 |
| `move_file` | 124 | 8.6 | 122.9 |
| `get_file_info` | 106 | 9.2 | 142.1 |

With a second breakpoint the rewritten segment `S` is the 389-token hot block; without
one it is the whole 11,641-token prefix. A 50-turn horizon can supply 8 uses of a tool
but not 98, so one provider clears the bar by 6× and the other misses by 2×. No
per-model tuning: the price ratio decides.

**Firing is not the same as paying off, and here it does not pay.** This is the only
configuration in the project where layer B does anything, so it is the one that decides
the idea — and `hotset` is dominated outright by `index-only`: 6.1 points worse strict
(p=0.018), 2.9 worse lenient (n.s.), and **22% more expensive** ($0.003184 vs
$0.002599). Against `static hot set` it is 11.3 points worse strict (p=0.000) at 20%
more cost. Three admitted schemas left it worse than shipping bare names for all 300.

The bill by cache class shows why, and it is not the write premium — HotSet writes
*fewer* tokens than `index-only` (94 vs 169) and still costs more:

| arm | uncached | cached | written | $/turn |
|---|---|---|---|---|
| index only | 548 | 12,934 | 169 | $0.002599 |
| static hot set | 554 | 15,378 | 22 | $0.002662 |
| **hotset (LRU-K)** | **1,179** | 13,299 | 94 | $0.003184 |
| full catalog | 525 | 44,719 | 437 | $0.006011 |

It reads 2.2× as many tokens at the uncached rate. Each mid-run admission invalidates
everything downstream of the tool block and the prefix re-warms; the hit-rate drop
(91.4% vs 94.7%) is that cost in aggregate. The re-warm amortises when an admitted tool
is called `n*` times — and on this suite nothing is called twice. The oracle predictor
admits nothing even on Haiku, so the honest reading is that **LRU-K is over-admitting
against its own break-even rule**, and the closed form is right in both directions:
right that admission is affordable here, right that these three tools do not clear it.

**Token reduction has a floor.** RAG-over-tools and lazy discovery post 0.9% and 0.0%
cache hit against 91–98% elsewhere. Haiku's minimum cacheable prefix is 4,096 tokens;
both arms build prefixes near or below it, so caching silently no-ops and every token
is billed uncached. RAG ends up costing more per correct call ($0.014254) than
`index-only` ($0.006888) while sending 5× fewer tokens.

**Strict accuracy is measuring the labels.** Twin selection runs 7.7–21.9% per arm and
is worst for HotSet, whose small hot set often holds one of a twin pair and not the
other: 31.6% strict against 53.5% lenient. Under lenient scoring HotSet is
indistinguishable from the full catalog (p=0.2624) at 47% of the token count; under
strict scoring it appears to lose badly (p=0.0003). Only the lenient comparison is
about routing.

**The frontier is one arm.** Lenient statistical dominance leaves `index-only` alone:
nothing is both cheaper and not-significantly-worse. Across a 2.3× cost range and a 3.3×
prompt-size range, five of six arms are statistically one arm on accuracy — the minimum
detectable gap at n=310 is 4.0%. Cost is the discriminator, again.

**No arm regresses mid-run.** `scripts/drift.py` tests each arm's loss rate against the
full catalog, per half, with Fisher exact; every arm comes back flat (p ≥ 0.364).

> Superseded: the earlier Haiku table used 95 held-out turns drawn by a `split()` that
> keyed on `hash((seed, scenario))`. CPython salts string hashing per process, so two
> sweeps shared only 6 of 95 labels and no cross-sweep comparison was valid. Fixed with
> `hashlib.blake2b` and a subprocess regression test; both sweeps above were re-run on
> the stable split at n=310. The break-even table was never affected — `n*` depends on
> token counts and prices, not on which turns were held out.

## Server-level skew cannot test admission, structurally

Every sweep above ran at `skew=0`, and the obvious repair — turn the skew knob up — does
not work. `trace()` weights arrivals per *server*, on the reasoning that a team adopts a
server and then uses all of it. But admission is priced per *tool*, and using all of a
server is still uniform at the tool level:

| skew | top-5 share | peak uses / 50 turns | busiest tool's rate | clears n*≈8 |
|---|---|---|---|---|
| 0.0 | 17.4% | 4 | 3.9% | no |
| 1.0 | 20.6% | 6 | 5.2% | no |
| 2.0 | 26.1% | 6 | 6.8% | no |
| 4.0 | 33.9% | 7 | 8.4% | no |

It saturates below the bar. The ceiling is structural: the harvested servers hold 13, 14,
12 and 24 tools, so even routing *100%* of traffic to `filesystem` puts its busiest tool
at 12.7% of turns, and to `git` at 15.1%. Asymptotically a tool earns its schema when its
call rate clears `c/u` — 10% on Haiku — so server skew brushes the bar only at infinite
concentration, and never reaches it in practice.

The missing variable is within-session tool reuse, which the frozen suite has almost
none of: **2 of 124 sessions repeat any tool.** The generation prompt asks for varied
servers and never asks for repetition, so every session is five distinct tools.

`repeat()` supplies it without inventing labels. A repeat turn is an existing labeled
turn from elsewhere in the suite that calls the same tool, inserted at a random position
in a session that already called it — so phrasing varies, ground truth stays exactly as
trustworthy as the suite it came from, and the session's own working set deepens rather
than widens. At `reuse=1.0` peak uses per 50-turn horizon goes 4 → 9, clearing `n*` for
the first time in the project.

## The forecast, not the price, is what loses the money

Admission fires on Haiku and loses. Two hypotheses fit that: the break-even rule is
wrong, or the rule is right and the predictor feeds it bad rates. `scripts/calibration.py`
separates them offline, at zero cost — the decision is a pure function of the trace, so
no model has to be called. For every tool the controller admits it records the uses the
predictor promised over the horizon against the uses the trace actually delivered in the
same 50-turn window, and swaps in `Oracle` to get the upper bound.

`uv run python scripts/calibration.py haiku`, 300 tools, held-out suite:

| reuse | peak/50 | predictor | admitted | predicted | actual | over-forecast | earned their schema |
|---|---|---|---|---|---|---|---|
| 0.0 | 4 | lru-k | 32 | 417.4 | 19 | 22.0x | **0 / 32** |
| 0.0 | 4 | oracle | 0 | 0.0 | 0 | — | 0 / 0 |
| 1.0 | 9 | lru-k | 62 | 1322.7 | 70 | 18.9x | 1 / 62 |
| 1.0 | 9 | oracle | 22 | 116.0 | 116 | 1.0x | **22 / 22** |
| 3.0 | 11 | lru-k | 66 | 1574.1 | 132 | 11.9x | 5 / 66 |
| 3.0 | 11 | oracle | 28 | 178.0 | 178 | 1.0x | 28 / 28 |

`earned` is scored against the floor threshold `T·c/u` = 5 uses, which only holds while
the hot set is empty; admitting raises the bar for everything after. So `earned` is an
upper bound on admissions that paid, and the true count is lower still.

Three things fall out.

**The economics are sound and the flat workload is genuinely inadmissible.** At
`reuse=0.0` the oracle — which counts the future directly — admits *nothing*. There is no
tool in the 310-turn suite whose real call rate clears 10%. The measured −6.1pt strict
and +22% cost against `index-only` is therefore not the price of cache-awareness. It is
32 admissions that a perfect forecaster would have declined, each one invalidating the
prefix and never repaying the rewrite.

**LRU-K over-forecasts by an order of magnitude, and does so structurally.**
`expected_uses = horizon · (k/span) · confidence` reads a backward K-distance as a rate
and extrapolates it linearly across the horizon. Two calls three turns apart imply 33
uses over 50 turns against a threshold of 5. Bursts are exactly what a five-turn session
produces, so the estimator's failure mode is aligned with the workload's shape rather
than random — which is why the error survives a 22x margin and does not average out over
310 turns.

**There is real headroom, and the current predictor cannot reach it.** At `reuse=1.0`
the oracle finds 22 tools that genuinely clear break-even; LRU-K finds 1 of them while
buying 61 that do not. The gap between those two rows is the entire value proposition of
layer B, and it is a prediction problem, not a caching one. That makes the week-4
predictor ablation the load-bearing experiment: a rate estimator that is merely
*calibrated* — even a poor one, so long as it is not biased 19x high — should recover
most of the oracle's 22 without needing to be accurate about which tools they are.

The `over-forecast` column for the oracle is 1.0x by construction: it counts the same
window it predicts. The informative column there is `admitted` — 0 → 22 → 28 across
reuse rates, which measures how much the workload leaves on the table before any
predictor is chosen.

## Reuse, added on purpose, and admission still never fires on a one-breakpoint provider

`repeat()` raised peak uses per 50-turn horizon from 4 to 9 — past the n*≈7.8–9.2 that
fires on Haiku. Re-running the week-3 sweep on qwen-flash over 620 turns:

| arm | strict | twin | lenient | halluc | hit | prompt tok | $/turn | $/correct |
|---|---|---|---|---|---|---|---|---|
| lazy-discovery | **56.0%** | 5.3% | 61.3% | 0.0% | 12.6% | 6,400 | $0.000186 | $0.000332 |
| static-hot-set | 44.7% | 13.7% | 58.4% | 2.7% | 96.8% | 13,941 | $0.000101 | **$0.000225** |
| rag-over-tools | 37.4% | 16.6% | 54.0% | 11.6% | 42.3% | 2,258 | **$0.000058** | $0.000156 |
| hotset | 35.2% | **31.1%** | **66.3%** | 2.4% | 92.5% | 12,559 | $0.000105 | $0.000299 |
| full-catalog | 34.8% | 29.2% | 64.0% | 5.3% | 98.4% | 39,384 | $0.000260 | $0.000745 |
| index-only | 32.9% | 19.2% | 52.1% | 7.1% | 96.4% | 11,974 | $0.000088 | $0.000268 |

**`hot set at end (0): []`.** Nothing was admitted, at any point, on any turn. The
workload cleared the bar the *other* provider sets and never came close to this one's.
With one cache breakpoint the rewritten segment is the entire 11,641-token prefix rather
than the hot block, which puts n* at 98–142 against a 50-turn horizon. Admission is
decided by the provider's cache architecture before the traffic is consulted at all.

At n=620 the detectable gap falls to 2.8%, so this is the best-powered sweep in the
project. `static-hot-set` beats `index-only` by **11.8 points strict** and
`full-catalog` by **9.9 points strict at 39% of the cost**.

## Schema presence is itself the disambiguation signal

The twin column above, read across the three schema regimes, explains every accuracy
result in the project:

| regime | who carries a schema | twin rate, haiku | twin rate, qwen |
|---|---|---|---|
| `index-only` | nobody | 19.7% | 19.2% |
| `static-K` | the real tool only | **3.2%** (K=64) | **13.7%** (K=16) |
| `hotset` | whatever BM25 retrieved, usually the twin | 21.9% | 31.1% |
| `full-catalog` | the tool *and* its twin | 14.8% | 29.2% |

When only the genuine tool carries a schema and its near-duplicate carries a bare name,
"has a schema" is evidence of being the genuine one, and twin confusion collapses. Give
both a schema and the signal disappears. Give it to the twin and not the target — which
is what layer C does on 98% of turns — and the signal is inverted.

That makes selective schema inclusion **better than complete inclusion**, not merely
cheaper, which is a stronger claim than the one the project set out to make. It also
predicts `full-catalog` must sit below `static-64` on strict accuracy.
