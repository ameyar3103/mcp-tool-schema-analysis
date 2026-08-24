# HotSet

A cache-aware tool router for MCP agents.

An agent's tool list renders *upstream* of its system prompt, so the prefix cache is
gated by the bytes of `tools`. Change one schema and every cached token downstream is
invalidated. HotSet treats the tool list as a cache-resident working set: a stable hot
block that earns its place, and a cold tail served on demand.

## The result

On qwen-flash over a 300-tool catalog and 310 held-out turns
([docs/pareto.md](docs/pareto.md)):

| arm | lenient acc | cache hit | prompt tok | $/turn |
|---|---|---|---|---|
| **hotset** | **61.9%** | 94.0% | 12,357 | $0.000099 |
| full-catalog | 57.7% | 98.5% | 39,193 | $0.000258 |

**3.2× fewer prompt tokens and 2.6× lower cost, with no accuracy loss to show for it**
(p=0.223). HotSet is on the statistical Pareto frontier; the full catalog is not.

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

## Layout

```
hotset/layout/     three-layer prompt assembly, byte-exact; token counting
hotset/policy/     baselines, the adaptive controller, predictors, break-even economics
hotset/eval/       runner, task suite, workload skew, significance tests, span traces
hotset/corpus/     harvested MCP tool catalog and synthetic near-duplicate distractors
hotset/runtime/    thin OpenRouter client — no agent framework, byte control is the point
```

## Running

```sh
uv sync --all-extras
uv run pytest -q                                   # offline; no key needed
uv run python scripts/simulate_admission.py haiku 300 2 0   # admission, no API calls
uv run python scripts/run_week3.py qwen-flash 300 2 0       # full sweep (needs a key)
uv run python scripts/plot_pareto.py <salt> out.png --lenient
uv run python scripts/drift.py <salt>
```

An OpenRouter key goes in a gitignored `.env`. Everything in CI runs without one.

## Reading the numbers

Two things about this harness are worth knowing before trusting any table it prints.

**Strict and lenient accuracy are both reported.** The catalog is padded with synthetic
near-duplicates of real tools, and the label blesses exactly one of each pair. Scoring a
twin as an ordinary error charges arms for a distinction their descriptions do not
support — it moved one comparison from p<0.001 to p=0.287. Strict cannot tell a wrong
tool from an equivalent one; lenient cannot tell a right tool from a lucky one.

**Every ranking carries its detection floor.** Under paired McNemar, power depends on
discordant pairs rather than sample size; at n=310 the minimum detectable gap is 4.0%.
The Pareto frontier is built on statistical dominance, and because "not significantly
worse" is not transitive, the frontier re-admits arms that a dominance chain would
otherwise drop.

## Caveats

- One flat workload, one tool call per turn, no tool repeated within a scenario, so
  per-tool call rates sit below the regime where admission should pay. The oracle
  predictor admits nothing, which is the controller being right rather than weak.
- Cross-provider accuracy comparisons need a shared held-out split; earlier sweeps did
  not have one and those claims were withdrawn.
- Twin provenance covers synthetic distractors only. Real near-duplicates across two MCP
  servers are still scored strictly, so lenient accuracy is a lower bound.
