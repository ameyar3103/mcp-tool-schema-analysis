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
