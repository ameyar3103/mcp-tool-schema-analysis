# Week 4 — Predictors, and when admission should refuse

Week 3 established the break-even. This asks the next question: given that admission
is priced correctly, does knowing *what follows what* change what gets admitted?

Every arm here is the same HotSet policy against the same economics. Only the
predictor differs, so any gap is attributable to prediction rather than layout.

## The threshold has a limit, and it is a price ratio

Taking `n* = S·(w−c)/(P·u) + T·c/u` and dividing by the horizon gives the *rate* a
tool must sustain:

```
n*/T  =  S·(w−c)/(P·u·T)  +  c/u   ⟶   c/u   as T grows
```

The rewrite term vanishes. What survives is the carrying cost: a cached token still
bills at `cached`, so a schema earns its place in the prefix **iff it is called more
often than the provider's cached/uncached price ratio.** Schema size and segment size
do not move this bound — they only decide how long the deployment must run before it
holds.

| model | `c/u` | rate needed at T=50 | T=200 | T=1000 | T=5000 |
|---|---|---|---|---|---|
| claude-haiku-4.5 | 0.100 | 0.169 | 0.117 | 0.103 | 0.101 |
| qwen3.7-flash | 0.200 | **2.110** | 0.678 | 0.296 | 0.219 |

The qwen row explains week 3's empty hot set without reference to any workload: at a
50-turn horizon a tool would have to be called 2.11 times *per turn* to earn a cached
schema. That is not a hard bar, it is an impossible one. Admission was never going to
fire, and no predictor could have changed that.

`amortization_horizon(spec, P, S, rate)` inverts this: how many turns of traffic before
admitting at a given rate pays for itself, and `inf` when the rate sits below `c/u`.

## The workload has no hot set, and the oracle proves it

The task suite is deliberately diverse — its generation prompt asks for varied servers
— so tool use is nearly flat. On the 310 held-out turns the top five tools take 14.5%
of traffic and the peak use in any 50-turn window is 4, against an `n*` of 34.

Replaying that trace offline through the controller (`scripts/simulate_admission.py`,
no API calls, ~5s):

| predictor | peak hot | admissions | rewrite $ | matches oracle? |
|---|---|---|---|---|
| **oracle** | 0 | **0** | $0.0000 | — |
| markov | 0 | **0** | $0.0000 | **yes** |
| ensemble | 4 | 28 | $0.0156 | no |
| lru-k | 7 | 50 | $0.0706 | no |

At skew 2.0 (Zipf arrival mix over servers, same sessions) the picture is unchanged:
oracle 0, markov 0, ensemble 23 admissions at $0.0202, LRU-K 46 at $0.0852.

This is the week's main result, and it is a negative one stated positively:

**Admission declining to fire is the controller being correct.** A perfect predictor,
handed the actual future, admits nothing — because no tool's true rate clears `c/u`
over a horizon long enough to amortize the write. Markov reproduces that decision
exactly. LRU-K disagrees fifty times and pays $0.071 in cache rewrites for it, roughly
9% on top of a $0.93-per-arm sweep, in exchange for nothing.

Without the oracle these two explanations are indistinguishable — "the predictor is
weak" and "admission never pays here" both show up as an empty hot set. That is the
whole reason it exists.

## Why LRU-K over-admits

LRU-K reads a rate off the backward K-distance: two calls within a short span implies
a high rate. Inside a five-turn scenario that is often true and briefly so. Projected
across a 310-turn horizon it is badly wrong, because the scenario ends and the tool is
never called again. Markov avoids this by construction: a transition is weighted by
`decay**t` and washes out into the marginal, which for a rarely-used tool is near zero.

The `peak` and `admissions` columns diverging — peak 7, admissions 50 — is the
signature. The hot set is not growing, it is churning, and every churn is a rewrite.

## Live ablation — qwen3.7-flash, 310 held-out turns (salt `804fb61f`)

Policy and economics held fixed; only the predictor varies. `index-only` is the
names-only floor, `hotset-oracle` the ceiling that sees the actual future.

| arm | strict | lenient | hit | prompt tok | hot | $/turn |
|---|---|---|---|---|---|---|
| hotset-oracle | 29.4% | **62.3%** | 94.3% | 12,341 | 0 | $0.000098 |
| hotset-ensemble | 31.0% | 61.0% | 94.2% | 12,344 | 0 | $0.000098 |
| hotset-lru-k | 29.0% | 59.4% | 94.0% | 12,357 | 0 | $0.000100 |
| hotset-markov | 31.9% | 59.4% | 94.3% | 12,340 | 0 | $0.000098 |
| index-only | 32.6% | 51.6% | 97.2% | 11,797 | 0 | $0.000086 |

**No predictor is distinguishable from any other, including the oracle.** Pairwise
lenient McNemar puts every predictor-vs-predictor comparison at p ≥ 0.163, against a
4.0% detection floor. All four beat `index-only` (p = 0.000–0.014) by 7.8–10.7 points.

That gap is not admission. The `hot` column is **0 for every arm** — on Alibaba `n*` is
98–142 and nothing clears it, so all four predictors produce the same plan. What the
ablation actually measures is the policy's layer-B suffix against a names-only prefix,
and the predictors are being compared on a decision none of them got to make.

This is the week-3 economics confirmed rather than a predictor result. A predictor
ablation on a workload where admission never fires has no signal in it by construction,
and reporting the four as "statistically tied" without that caveat would imply the
predictors were tested and found equivalent. They were not tested at all.

The offline replay in the section above is where predictor quality does separate: LRU-K
admits 50 tools for $0.071 of rewrites that the oracle shows were never worth buying.
Those admissions are invisible here because the live arm ran against a provider whose
threshold none of them reached.

## What this does not show

The suite gives one tool call per turn and never repeats a tool inside a scenario, so
per-tool rates are bounded by session composition. Real agent traffic repeats tools
heavily within a session (`read_file` a dozen times while editing), which is exactly
the regime where admission should pay. Skew over *servers* cannot manufacture that;
it needs traces with intra-session repetition. Until then the honest claim is narrow:
on flat workloads the controller correctly refuses, and the predictor that refuses
with it is the better one.
