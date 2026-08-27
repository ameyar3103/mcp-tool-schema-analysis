# Week 5 — The selector was wrong: budget, not break-even

Week 3 shipped an admission controller that fired on Haiku and lost: `hotset` came in
6.1 points worse strict than `index-only` (p=0.018) at 22% more cost. Weeks 3–4 blamed
the workload. That was only a third of the story. Three offline replays — none of which
call a model, because admission is a pure function of the trace — locate the rest.

## 1. Break-even has no term for accuracy

The closed form asks whether caching a schema is cheaper than tail-loading it, which
assumes the two are substitutes. They are not. `static-hot-set` carries 16 schemas that
the threshold explicitly rejects, and beats `index-only` by **5.2 points strict** on
Haiku and **11.8 points strict** on qwen under reuse. A cached schema buys accuracy; the
cost model prices only tokens, so it optimises `$/turn` when the objective is
`$/correct`.

This is the actual defect in the idea as written. The threshold answers *"is this schema
free?"* — a real question, correctly answered. It cannot answer *"is this schema worth
paying for?"*, which is the question a deployment has.

## 2. Churn, not caching, is where the money went

Every membership change rewrites layer B and re-warms the prefix downstream of it. Under
the week-3 configuration the hot set changed on **65 of 310 turns** — one turn in five —
to hold a set averaging 2.0 tools.

| epoch | rewrites | tokens written | cost of the rewrites |
|---|---|---|---|
| 1 (as shipped) | 65 | 21,562 | $0.02695 |
| 10 | 22 | 6,318 | $0.00790 |
| 25 | 12 | 2,022 | $0.00253 |
| 50 | 6 | 1,319 | $0.00165 |

Batching admissions into epochs makes N admissions share one rewrite. It costs nothing
in responsiveness that matters: the tail is still recomputed every turn, only membership
is frozen.

## 3. Layer C is a distractor amplifier

Layer C tail-loads BM25's top-k for the current query. On a catalog padded with
near-duplicate twins, that is close to the worst possible thing to do:

| | share of turns |
|---|---|
| tail contains a **twin's** full schema | **98.1%** |
| tail contains the **target's** full schema | **36.8%** |

The retriever finds the impostor more reliably than the real tool, and then hands it the
one thing that distinguishes them. This is why `hotset`'s twin rate (21.9%) exceeds
`index-only`'s (18.7%) despite `hotset` showing strictly more information. On a
twin-heavy catalog, `tail_k=0` is the correct setting.

## 4. Adaptivity is a liability on stationary traffic

Matched budget, so the only difference is who picks the members. Coverage is the share of
held-out calls whose target sits in the hot set:

| K | static (train frequency) | adaptive (LRU-K) |
|---|---|---|
| 8 | **14.8%** | 10.3% |
| 16 | **27.7%** | 20.3% |
| 32 | **55.2%** | 41.9% |
| 64 | **90.6%** | 75.8% |

The training prior loses to nothing and beats LRU-K everywhere. The traffic is
stationary, so past frequency *is* the estimate; adaptivity only adds variance. A
calibrated rate estimator does not rescue this — `Rate` cuts LRU-K's 18.9x
over-forecast to 5x and then admits nothing at all, because the profitable admissions
are bursts and a burst is not identifiable until it is over.

## The experiment

Given all four, the controlled variable is the schema budget and membership is fixed for
the run. `scripts/run_week5.py` sweeps K from 0 (`index-only`) to 300 (`full-catalog`),
with `adaptive-64` at the same budget as `static-64` so any gap between them is
prediction rather than spend.

Coverage of 90.6% at K=64 costs 9,585 schema tokens against the full catalog's ~45,681
prompt tokens per turn, so the frontier is expected to bend well short of K=300.

<!-- RESULTS -->
