# Week 6 — the frontier, and what the suite can actually support

Run `97b45e2a`: qwen-flash, 300-tool catalog (corpus v2), 62 held-out sessions / 310
turns, flat workload. Every arm shares the salt, so none inherited a warm prefix.

| arm | strict | lenient | cache hit | prompt tok | $/turn | $/lenient-correct | hops |
|---|---|---|---|---|---|---|---|
| rag-over-tools | 33.2% | 52.3% | 5.1% | 2,060 | $0.000080 | $0.000154 | 0.00 |
| index-only | 31.6% | 54.5% | 97.2% | 11,793 | $0.000086 | $0.000157 | 0.00 |
| static-hot-set | 43.5% | 58.1% | 97.4% | 13,769 | $0.000099 | $0.000170 | 0.00 |
| **hotset** | 31.0% | **61.9%** | 94.0% | 12,357 | $0.000099 | $0.000160 | 0.00 |
| lazy-discovery | 55.8% | **61.9%** | 15.7% | 4,262 | $0.000123 | $0.000199 | 0.96 |
| full-catalog | 36.5% | 57.7% | 98.5% | 39,193 | $0.000258 | $0.000448 | 0.00 |

**Statistical frontier (lenient): `rag-over-tools`, `hotset`, `lazy-discovery`.**

Read that frontier carefully, because it flatters the wrong thing. `hotset` is on it, but
on qwen-flash **the hot set is empty** — n* is 98–142 calls against a horizon that offers
one, so admission correctly refuses every tool. The arm labelled `hotset` here is layer A
plus a BM25 suffix; the controller contributed nothing. And it is on the frontier only
under lenient scoring: strict, it is 31.0%, second-worst in the table, behind
`static-hot-set` by 12.6pt (p=0.000) and `lazy-discovery` by 24.8pt (p=0.000).

The comparison that would justify the project is `hotset` against the *cheap* baselines,
not against `full-catalog` — which is the most expensive arm here and beats nothing.
Against `index-only` (+7.4pt lenient, p=0.015) and `rag-over-tools` (+9.7pt, p=0.003)
HotSet wins; against `static-hot-set` at the same $/turn it does not. With an empty hot
set, none of those deltas measure admission.

## Strict scoring measures the label, not the router

The strict column says `lazy-discovery` beats `full-catalog` by 19.3 points at p<0.001,
which reads as "a 300-tool prompt degrades routing." Counting a synthetic twin as
correct puts the same comparison at **p=0.287**, and every other significant strict
difference disappears with it. Only `lazy-discovery` over `rag-over-tools` survives.

The deficit was twin selection. Arms that hold more of the catalog hold more
near-duplicates, and the label blesses exactly one of each pair, so strict accuracy
charges them for a distinction their descriptions do not support. HotSet is the extreme
case: 31.0% strict against 61.9% lenient, a 30.9-point gap, because a small hot set
often contains one of a twin pair and not the other.

Neither view is sufficient alone. Strict cannot tell a wrong tool from an equivalent
one; lenient cannot tell a right tool from a lucky one. Both are reported.

## Dominance is not transitive

The frontier rule is: an arm is dominated when something cheaper is **not significantly
worse** under paired McNemar. That puts the burden of proof on the accuracy claim rather
than the cost claim, which is the right direction here — cost is arithmetic on token
counts, accuracy is an estimate with a 4.0-point detection floor.

Applied naively it produced a wrong answer. `rag-over-tools` ties `index-only`, which
ties `hotset`, so `hotset` fell off the frontier while beating `rag-over-tools` at
p=0.0032. Every pairwise step was correct; the recommendation was not. Two gaps below
the detection floor chain into one above it, so "not significantly worse" is not an
equivalence relation and the antichain is not the frontier.

`frontier()` therefore re-admits any arm that significantly beats a survivor, iterated
to a fixpoint. `inversions()` is the guard: on the reported set it must come back empty.

## The suite has power now, and the arms still mostly tie

At n=310 the minimum detectable gap is 4.0%, down from 7.2% at n=95. That was enough to
expose the twin artifact — at n=95 it was invisible. It is still not enough to separate
most arms on lenient accuracy, and that is the result rather than a shortfall: across a
3.2× cost range and a 19× prompt-size range, accuracy is flat. Cost is the discriminator.

## Behaviour does not drift, and late turns are easier

`scripts/drift.py` reads the saved traces two ways. Across the deployment, first half
against second half, a DRIFT verdict is issued only when the Wilson intervals separate;
`index-only` came back **flat** (31.0% vs 34.2%).

Within a conversation the pattern is the opposite of the feared one. Accuracy rises with
turn index — 26%, 24%, 31%, 40%, 42% for turns 0 through 4 — because earlier tool calls
sit in the history and disambiguate the twins. Compaction does not degrade late turns;
turn 0 is simply the hardest, being the one with no context to disambiguate against.

## Haiku, where admission actually fires — and loses

Run `52cbabb1`, same catalog and same 310 held-out turns, on `claude-haiku-4.5`. Anthropic
honours a second cache breakpoint, so `S` is the 389-token hot block instead of the whole
prefix, n* drops to 7.8-9.2, and LRU-K admits three tools. This is the only configuration
in the project where layer B does anything, so it is the one that decides whether the
idea works.

| arm | strict | twin | lenient | cache hit | prompt tok | $/turn | $/lenient-correct |
|---|---|---|---|---|---|---|---|
| index-only | 37.7% | 18.7% | **56.5%** | 94.7% | 13,651 | **$0.002599** | **$0.004605** |
| static-hot-set | **42.9%** | 11.9% | 54.8% | 96.4% | 15,954 | $0.002662 | $0.004853 |
| **hotset** | 31.6% | 21.9% | 53.5% | 91.4% | 14,572 | $0.003184 | $0.005945 |
| rag-over-tools | 22.6% | 8.4% | 31.0% | 0.9% | 2,531 | $0.003219 | $0.010394 |
| lazy-discovery | 36.8% | 7.7% | 44.5% | 0.0% | 5,130 | $0.005976 | $0.013424 |
| full-catalog | 41.6% | 14.8% | **56.5%** | 97.9% | 45,681 | $0.006011 | $0.010647 |

`hotset` is dominated outright by `index-only`: **6.1 points worse strict (p=0.018)**,
2.9 worse lenient (n.s.), and **22% more expensive**. It is 11.3 points worse than
`static-hot-set` strict (p=0.000) at 20% more cost. Three admitted schemas made it worse
than shipping names for all 300 tools.

Splitting the bill by cache class shows the mechanism, and it is not the write premium:

| arm | uncached | cached | written | $/turn |
|---|---|---|---|---|
| index-only | 548 | 12,934 | 169 | $0.002599 |
| static-hot-set | 554 | 15,378 | 22 | $0.002662 |
| **hotset** | **1,179** | 13,299 | 94 | $0.003184 |
| full-catalog | 525 | 44,719 | 437 | $0.006011 |

HotSet writes *fewer* tokens than `index-only` (94 vs 169) and still costs more, because
it reads 2.2× as many at the uncached rate. Each mid-run admission invalidates everything
downstream of the tool block and the prefix has to re-warm; the hit rate drop (91.4% vs
94.7%) is that cost showing up in the aggregate. On traffic where an admitted tool is
called n* times the re-warm amortises. Here nothing is called twice, so it never does —
which is precisely what the oracle predictor says by admitting nothing at all.

The conclusion this run supports is narrow and negative: **layer A pays, layer B does
not, and the shipped LRU-K predictor is over-admitting relative to its own economics.**

## What this does not show

- Two models, one flat workload. Both frontiers are shaped by that workload, and it is
  the wrong shape for the thing under test — 620 turns from 124 generated sessions, one
  tool call per turn, no tool repeated within a scenario. Per-tool call rates are bounded
  below the regime where admission pays, by construction of the suite.
- So the negative result above is scoped: it says LRU-K over-admits on traffic with no
  reuse, not that cache-aware admission fails in general. Testing the general claim needs
  a workload with intra-session tool repetition, which this suite does not contain.
- The strict/lenient split is doing a lot of work in every ranking here. `hotset` moves
  from second-worst to frontier depending on which one is used; treat single-column
  comparisons of this table as unsupported.
- Twin provenance covers synthetic distractors only. Genuine near-duplicates between two
  real MCP servers are scored strictly, so the lenient column is a lower bound.

## Replication: how much of a gap is just sampling?

Two sweeps ran the identical configuration — qwen-flash, corpus v2, 300 tools, skew 0,
no reuse — under different salts (`97b45e2a`, `421efa26`). Nothing differed but the run
nonce and the provider's sampling:

| arm | run A | run B | delta | McNemar p |
|---|---|---|---|---|
| full-catalog | 57.7% | 59.0% | +1.3 | 0.7122 |
| hotset | 61.9% | 61.3% | −0.6 | 0.8555 |
| **index-only** | 54.5% | 49.7% | **−4.8** | 0.0959 |
| lazy-discovery | 61.9% | 60.3% | −1.6 | 0.5831 |
| rag-over-tools | 52.3% | 51.3% | −1.0 | 0.7660 |
| static-hot-set | 58.1% | 56.1% | −1.9 | 0.5446 |

Five of six arms replicate within 2 points. One moved 4.8 — larger than the 4.0%
detection floor that the same table quotes for distinguishing two *arms*. The two
quantities are not the same thing: the floor describes a paired within-run comparison,
and this is between-run stability of one arm under resampling.

The cause was that `pinned_body()` never set `temperature`, so every request used the
provider default. It is pinned to 0 now, which removes our contribution to the variance
but not the provider's. **Every sweep reported in this repo predates that pin**, so read
single-run gaps of roughly 5 points or less as provisional, including the −6.1pt strict
result for `hotset` against `index-only` on Haiku. Gaps established at p<0.001 across
hundreds of discordant pairs are not at risk; the marginal ones are.
