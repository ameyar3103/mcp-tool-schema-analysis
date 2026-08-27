# HotSet

A cache-aware tool router for MCP agents.

An agent's tool list renders *upstream* of its system prompt, so the prefix cache is
gated by the bytes of `tools`. Change one schema and every cached token downstream is
invalidated. HotSet treats the tool list as a cache-resident working set: a stable hot
block that earns its place, and a cold tail served on demand.

## The result

Two findings, one positive and one negative. The negative one is about the thing this
repo is named after.

**Dropping every JSON Schema is free.** On Haiku over a 300-tool catalog and 310 held-out
turns, `index-only` (tool names, no schemas) and `full-catalog` both score 175/310 lenient
— 16 discordant pairs each way, **p = 1.0000**. An exact tie at 30% of the prompt tokens
and 43% of the cost:

| model | arm | lenient acc | prompt tok | $/turn |
|---|---|---|---|---|
| haiku-4.5 | **index-only** | 56.5% | 13,651 | **$0.002599** |
| haiku-4.5 | full-catalog | 56.5% | 45,681 | $0.006011 |

**Cache-aware admission does not pay on this workload.** Where admission actually fired
(Haiku, 3 tools admitted), `hotset` is *worse and more expensive* than simply shipping
names for the whole catalog:

| haiku-4.5 | strict | lenient | $/turn | vs index-only |
|---|---|---|---|---|
| index-only | 37.7% | 56.5% | $0.002599 | — |
| static-hot-set | 42.9% | 54.8% | $0.002662 | +5.2pt strict |
| **hotset** | **31.6%** | 53.5% | **$0.003184** | **−6.1pt strict (p=0.018), +22% cost** |

The cost penalty is not the cache-write premium — HotSet writes *fewer* tokens than
`index-only` (94 vs 169). It is prefix invalidation: 1,179 uncached tokens per turn
against `index-only`'s 548. Every mid-run admission re-warms the prefix, and three
admitted schemas never earn that back. This is the live confirmation of the offline
finding that the LRU-K predictor over-admits relative to the oracle — and on this
workload the oracle admits nothing at all.

The economics is still right; the workload is what's missing. See
[Why admission loses here](#why-admission-loses-here).


The admission rule is a closed form, not a tuned heuristic. A tool earns a cached schema
when its expected call rate over the horizon clears

```
n* = S·(w−c)/(P·u) + T·c/u        →   n*/T → c/u
```

where `S` is the rewritten segment, `P` the schema size, and `u`/`c`/`w` the uncached,
cached and write per-token prices. Asymptotically the schema size drops out: **a tool
earns its place iff its call rate exceeds the provider's cached/uncached price ratio.**
That single number explains why admission fires on Anthropic (n*≈8) and never on Alibaba
(n*≈98–142) for the same catalog and the same traffic — Anthropic honours a second cache
breakpoint, so `S` is the hot block (389 tok) rather than the whole prefix (11,641).
See [docs/admission.md](docs/admission.md).

## Why admission loses here

The break-even rule is a statement about *call rates*, and this suite does not produce
them. Each scenario calls one tool per turn and never calls the same tool twice, so the
empirical per-tool rate over a session sits far below `n*` for every tool in the catalog.
The oracle predictor — which sees the future and admits whatever clears break-even —
admits **nothing**. LRU-K admits three, so LRU-K is wrong, and the live Haiku numbers are
exactly the price of being wrong: three prefix re-warms nobody asked for.

That makes this a clean negative result rather than a bug. Admission is a bet that a tool
will be called often enough to amortise its cached schema; on bursty, repetitive traffic
the bet is good, and on flat single-shot traffic there is no bet to make. The honest
scope claim is: **layer A (drop the schemas) is where the money is, and layer B/C need a
workload with intra-session tool reuse before they can be evaluated at all.**

The week-4 predictor ablation inherits this. On qwen the hot set is empty for all four
predictors, so oracle, ensemble, markov and LRU-K emit byte-identical plans and are
indistinguishable (p >= 0.163) — a null by construction, not evidence of parity.

## Layout

```
hotset/layout/     three-layer prompt assembly, byte-exact; token counting
hotset/policy/     baselines, the adaptive controller, predictors, break-even economics
hotset/eval/       runner, task suite, workload skew, significance tests, span traces
hotset/corpus/     harvested MCP tool catalog and synthetic near-duplicate distractors
hotset/runtime/    thin OpenRouter client — no agent framework, byte control is the point
```

| doc | what it settles |
|---|---|
| [docs/admission.md](docs/admission.md) | the break-even derivation, and both live sweeps |
| [docs/pareto.md](docs/pareto.md) | the frontier, and why HotSet is not on the Haiku one |
| [docs/corpus.md](docs/corpus.md) | distractor design moves strict accuracy by 9 points |
| [docs/predictors.md](docs/predictors.md) | oracle vs LRU-K vs markov vs ensemble |
| [docs/baselines.md](docs/baselines.md) | what each arm actually sends |
| [docs/week1-findings.md](docs/week1-findings.md) | the cache-position measurements this started from |

## Running

```sh
uv sync --all-extras
uv run pytest -q                                   # offline; no key needed
uv run python scripts/simulate_admission.py haiku 300 2 0   # admission, no API calls
uv run python scripts/run_week3.py qwen-flash 300 2 0       # full sweep (needs a key)
uv run python scripts/plot_pareto.py <salt> out.png --lenient
uv run python scripts/drift.py <salt>
uv run python scripts/twin_exposure.py qwen-flash 300 2   # plan replay, no API calls
```

An OpenRouter key goes in a gitignored `.env`. Everything in CI runs without one.

## Reading the numbers

Three things about this harness are worth knowing before trusting any table it prints.

**Strict and lenient accuracy are both reported.** The catalog is padded with synthetic
near-duplicates of real tools, and the label blesses exactly one of each pair. Scoring a
twin as an ordinary error charges arms for a distinction their descriptions do not
support — it moved one comparison from p<0.001 to p=0.287. Strict cannot tell a wrong
tool from an equivalent one; lenient cannot tell a right tool from a lucky one.

**Strict accuracy is a property of the corpus, not just the router.** Re-running the
same 310 turns against the same target tools, changing only the disambiguating clause on
synthetic distractors, cost `hotset` **11.6 points of strict accuracy** while its lenient
accuracy moved 1.0 — and an offline plan replay confirms the arm's exposure to the
target's twin was identical across both runs. Twin confusion rose for all six arms, and
the sign of the strict change is not even consistent: `full-catalog` gained 3.2 points
over the same treatment. See [docs/corpus.md](docs/corpus.md).

**Single-run gaps of ~5 points are provisional.** Two sweeps of an identical config
under different salts agreed within 2 points on five of six arms and differed by 4.8 on
the sixth — above the detection floor. `temperature` went unpinned until after every
sweep reported here; it is 0 now. Gaps at p<0.001 are unaffected, marginal ones are not.

**Every ranking carries its detection floor.** Under paired McNemar, power depends on
discordant pairs rather than sample size; at n=310 the minimum detectable gap is 4.0%.
The Pareto frontier is built on statistical dominance, and because "not significantly
worse" is not transitive, the frontier re-admits arms that a dominance chain would
otherwise drop.

## Caveats

- One flat workload, one tool call per turn, no tool repeated within a scenario, so
  per-tool call rates sit below the regime where admission should pay. The oracle admits
  nothing; the shipped LRU-K predictor admits three and loses money for it. Nothing here
  says admission cannot work — only that it is untested on traffic that would reward it.
- Cross-provider accuracy comparisons need a shared held-out split; earlier sweeps did
  not have one and those claims were withdrawn.
- Twin provenance covers synthetic distractors only. Real near-duplicates across two MCP
  servers are still scored strictly, so lenient accuracy is a lower bound.
