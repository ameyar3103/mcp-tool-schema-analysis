# HotSet

**A cache-aware tool router for MCP agents — and a measurement that the schema budget is
an accuracy knob with an interior optimum.**

An agent's tool list renders *upstream* of its system prompt, so the prefix cache is gated
by the bytes of `tools`: change one schema and every cached token downstream of it is
invalidated. HotSet treats the tool list as a cache-resident working set — a stable hot
block of full schemas, over a whole-catalog index of bare names, with an optional
per-turn tail.

The project was built to answer a cost question. The answer it produced is about accuracy.

## The result

**64 of 300 tool schemas beat all 300 by 13.2 points strict, at 56% of the cost.**

haiku-4.5, a 300-tool catalog padded with near-duplicates, 310 held-out turns, one salted
run, paired McNemar with a 4.0% minimum detectable gap:

| K cached | arm | strict | twin | lenient | $/turn | $/correct |
|---|---|---|---|---|---|---|
| 0 | index-only | 38.4% | 19.7% | 58.1% | **$0.002597** | $0.006765 |
| 16 | static-16 | 43.2% | 15.2% | 58.4% | $0.002662 | **$0.006158** |
| 32 | static-32 | 44.5% | 9.0% | 53.5% | $0.002943 | $0.006612 |
| **64** | **static-64** | **52.9%** | **3.2%** | 56.1% | $0.003463 | $0.006546 |
| 64 | adaptive-64 | 41.6% | 12.9% | 54.5% | $0.003421 | $0.008222 |
| 300 | full-catalog | 39.7% | 17.1% | 56.8% | $0.006186 | $0.015590 |

![budget frontier](docs/assets/frontier.png)

Three findings, ordered by how much they should change what you build.

**The schema budget has an interior optimum.** Strict accuracy runs 38.4 → 43.2 → 44.5 →
**52.9** → 39.7 as K goes 0 → 16 → 32 → 64 → 300. The peak sits at 21% of the catalog.
`static-64` beats `full-catalog` 49/8 (p<0.001) and `index-only` 55/10 (p<0.001). Both
endpoints — send everything, send nothing — are dominated by the middle. This reproduces
the tool-overload failure the project set out to address, on our own catalog rather than
by citation, and locates the peak rather than just asserting one exists.

**Sending every schema is statistically indistinguishable from sending none.**
`full-catalog` vs `index-only` is 21/17, **p = 0.627**, while costing 2.4x more. The
industry default and the cheapest possible arm are the same arm, measured.

**The gain is entirely near-duplicate disambiguation.** Lenient accuracy never moves
(58.1 → 56.8; `static-64` vs `index-only` p=0.238) while the twin rate traces a clean V:
19.7 → 15.2 → 9.0 → **3.2** → 17.1. A bare name already puts the model in the right
neighbourhood. What a name cannot do is separate a tool from a synthetic twin that differs
only in its argument schema.

## Why this may mean more than the table shows

**A cached schema is bought as evidence of authenticity, not as information.** This is the
mechanism behind every number above, and it is a claim about *asymmetry*, not about
budget. When only one member of a confusable pair carries a schema, "has a schema" is
itself a discriminative feature — the model does not have to read the schema to use it.
That predicts the non-monotonicity exactly: at K=300 the twin gets a schema too, the
signal is destroyed, and twin confusion snaps back from 3.2% to 17.1%. The curve is not a
context-length effect; it is a signal-to-noise effect that a longer context window will
not fix.

**It inverts the usual advice.** The standard response to a large tool catalog is
retrieval: fetch the relevant schemas per turn. Here that is the *worst* move — replaying
the plans offline, BM25's tail puts some near-duplicate's full schema in the prompt on
**98.1%** of turns, and the target's *own* twin on 25.6% against the target's 38.3%. The
retriever is nearly as likely to arm the impostor as the real tool. On a catalog with
near-duplicates the per-turn tail should be off and the schemas chosen ahead of time.

**It relocates the benchmark problem.** If `full-catalog` and `index-only` are one arm at
p=0.627, then any tool-selection result reported only against "send the whole catalog" is
reported against a baseline that does nothing — including several of this project's own
earlier results. The cheap baseline is the one that has to be beaten.

**The scope is honest but narrow.** One provider pair, one 300-tool catalog, one flat
workload, and the twins are synthetic. What generalises is the mechanism and the method:
sweep the budget, score strict *and* lenient, and report the detection floor. What does
not yet generalise is the location of the peak at 21%.

## Technical details

### Three-layer prompt

Layers A and B render as **system text blocks**, never the native `tools` field, because
only text blocks honour an explicit cache breakpoint on OpenRouter.

| layer | content | cached | invalidated by |
|---|---|---|---|
| A — `## CATALOG` | every tool as `name(arg, optional?) - summary` | yes | catalog change only |
| B — `## SCHEMAS` | full JSON Schema for admitted tools | yes | a membership change |
| C — `## ADDITIONAL SCHEMAS` | this turn's retrieved schemas, after history | no | every turn |

Layer A is what makes a partial budget safe: a tool outside the hot set is still nameable,
so trimming schemas does not trim reach. Layer B is name-sorted so admission *order* never
shifts bytes. Layer C is appended after the conversation history so it cannot invalidate
anything upstream — and on a twin-heavy catalog it should be empty.

### Policies

`hotset/policy/` — every arm is a `Policy`: `plan(catalog, history, query) -> Plan`.

- `IndexOnly` / `FullCatalog` — the K=0 and K=300 endpoints.
- `StaticHotSet` + `frequency_hot_set` — the winner. Ranks tools by training-set call
  frequency, takes the top K, never changes.
- `HotSet` — the adaptive controller. In `budget` mode it takes the predictor's top K; in
  threshold mode it admits whatever clears break-even. `epoch` batches membership changes
  so N admissions share one prefix rewrite; `tail_k` sizes layer C.
- `RagOverTools` / `LazyDiscovery` — retrieval and MCP-Zero-style reference arms.

`hotset/policy/economics.py` holds the closed form. Admission pays after

```
n* = S·(w − c)/(P·u) + T·c/u          n*/T → c/u  as T → ∞
```

where `S` is the rewritten segment, `P` the schema tokens, and `u, c, w` the uncached,
cached and write prices. The asymptotic floor `c/u` is the term the original formulation
omitted: cached tokens still bill, so no horizon makes a schema free. `S` is
provider-dependent — Anthropic honours a second breakpoint (S=389, n*≈8), Alibaba rewrites
the whole prefix (S=11,641, n*≈98–142) — and that prediction held on both live sweeps.

### Evaluation

`hotset/eval/` — 62 held-out multi-turn sessions, 310 turns, one labelled tool call each,
split by `blake2b` so the split is stable across processes. Every request is
provider-pinned with `temperature=0`, and the prompt leads with a per-run salt so a
re-run inside the cache TTL cannot inherit a prefix it never paid for.

Three scores, always reported together:

- **strict** — the labelled tool, exactly.
- **twin** — a synthetic near-duplicate of the labelled tool. Tracked via `twin_of`
  provenance rather than scored as an ordinary error.
- **lenient** — strict or twin. Strict cannot tell a wrong tool from an equivalent one;
  lenient cannot tell a right tool from a lucky one.

Rankings carry their detection floor. Under paired McNemar power depends on *discordant*
pairs, not sample size; at n=310 the floor is 4.0%. The Pareto frontier is built on
statistical dominance — an arm is dropped only when something cheaper is not
*significantly* worse — and because "not significantly worse" is not transitive, the
frontier re-admits any arm that measurably beats a survivor, to a fixpoint.

![pareto frontier](docs/assets/pareto.png)

### Corpus

76 real tools harvested from 8 public MCP servers, padded to 300 with synthetic
near-duplicates that share a target's domain and argument shape and differ in one
disambiguating clause. The padding is the experiment, not scenery: re-running the same 310
turns while changing only that clause moved one arm's strict accuracy by −11.6 points and
another's by +3.2, with lenient accuracy nearly flat.

### Runtime

`hotset/runtime/openrouter.py` — a ~150-line client over Chat Completions. No agent
framework: the experiment needs byte-exact control of the prompt, and a framework that
reorders or re-renders blocks would silently change the quantity being measured. Requests
are pinned to a single provider with `allow_fallbacks: false`, since a silent failover
moves the cache.

## Layout

```
hotset/layout/     three-layer prompt assembly, byte-exact; token counting
hotset/policy/     baselines, the adaptive controller, predictors, break-even economics
hotset/eval/       runner, task suite, workload skew, significance tests, span traces
hotset/corpus/     harvested MCP catalog and synthetic near-duplicate distractors
hotset/runtime/    thin OpenRouter client
```

[docs/findings.md](docs/findings.md) has the full derivation, the offline diagnostics that
located the failure, and every table.

## Running

```sh
uv sync --all-extras
uv run pytest -q                                       # offline; no key needed
uv run python scripts/calibration.py haiku 300 2       # admission replay, no API calls
uv run python scripts/twin_exposure.py haiku 300 2     # plan replay, no API calls
uv run python scripts/run_sweep.py frontier haiku 300 2 0    # the paid sweep (needs a key)
uv run python scripts/run_sweep.py baselines haiku 300 2 0
uv run python scripts/plot_frontier.py <salt>
uv run python scripts/plot_pareto.py <salt> pareto.png
uv run python -m hotset.cache.probe haiku              # provider cache probes (needs a key)
```

An OpenRouter key goes in a gitignored `.env`. Everything in CI runs without one.

## Caveats

- One flat workload with one tool call per turn and no tool repeated within a scenario, so
  per-tool call rates sit below the regime where admission should pay. Under an oracle
  predictor at `reuse=1.0` there are 22 genuinely profitable admissions; LRU-K finds 1 of
  them while buying 61 that lose money. Nothing here says admission cannot work — only
  that it is untested on traffic that would reward it.
- Single-run gaps of ~5 points are provisional. `temperature` went unpinned until after
  the earlier sweeps; two runs of one arm then differed by 4.8 points, above the floor.
  It is pinned to 0 now, and gaps at p<0.001 are unaffected.
- Twin provenance covers synthetic distractors only. Real near-duplicates across two MCP
  servers are still scored strictly, so lenient accuracy is a lower bound.
- Cross-provider accuracy comparisons need a shared held-out split; earlier sweeps did not
  have one and those claims were withdrawn.
