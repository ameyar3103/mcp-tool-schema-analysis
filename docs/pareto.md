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
HotSet sits on it at 3.2× fewer prompt tokens and 2.6× lower cost than the full
catalog, with no accuracy loss to show for it (61.9% vs 57.7%, p=0.223).

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

## What this does not show

- One model (qwen-flash) and one flat workload. The Haiku frontier is not this frontier:
  the two providers differ on split cache breakpoints, which is what decides admission.
- 620 turns from 124 generated sessions, one tool call per turn, no tool repeated within
  a scenario. Per-tool call rates are bounded below the regime where admission pays.
- Twin provenance covers synthetic distractors only. Genuine near-duplicates between two
  real MCP servers are scored strictly, so the lenient column is a lower bound.
