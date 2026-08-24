# Distractor design is a treatment, not a detail

The catalog holds 76 tools harvested from real MCP servers, padded to 300 with synthetic
near-duplicates. A distractor clones a real tool under a confusable name, keeps its
schema verbatim, and appends one disambiguating clause to the description. That clause is
the only thing that separates corpus v1 from v2:

- **v1** appends a generic clause from a fixed pool — *"Results are not paginated."*,
  *"Runs without acquiring a lock."*
- **v2** appends a clause drawn from the source server's own domain, falling back to the
  generic pool only for servers without one.

Everything else is held fixed. Both runs are qwen-flash over 300 tools, and the split is
keyed on `blake2b(seed, scenario)` rather than the corpus, so the two sweeps share **all
310 held-out turns and all 310 target tools** — verified, not assumed. The only free
variable is the clause.

## v2 made twins harder to tell apart, without making them more visible

v2 was written to make synthetics *more* distinguishable. It did the opposite. Twin
confusion rose for all six arms:

| arm | v1 twin | v2 twin | delta | Fisher p |
|---|---|---|---|---|
| static-hot-set | 13.5% | 14.5% | +1.0 | 0.8172 |
| lazy-discovery | 2.9% | 6.1% | +3.2 | 0.0801 |
| rag-over-tools | 15.5% | 19.0% | +3.5 | 0.2879 |
| index-only | 15.8% | 22.9% | **+7.1** | **0.0325** |
| hotset | 20.3% | 31.0% | **+10.6** | **0.0032** |
| full-catalog | 10.3% | 21.3% | **+11.0** | **0.0003** |

Replaying every arm's plan offline against both catalogs (`scripts/twin_exposure.py`,
no API calls) shows the increase is not an
exposure artifact. How often each arm puts the *target's own twin* in a schema-bearing
layer barely moves between versions, and for four of six arms it is identical by
construction:

| arm | twin schema shown, v1 | v2 | target's own schema shown, v1 | v2 |
|---|---|---|---|---|
| full-catalog | 100.0% | 100.0% | 100.0% | 100.0% |
| rag-over-tools | 52.9% | 51.3% | 54.2% | 51.9% |
| hotset | 24.9% | 25.6% | 44.1% | 38.3% |
| static-hot-set | 0.0% | 0.0% | 25.3% | 27.9% |
| index-only | 0.0% | 0.0% | 0.0% | 0.0% |
| lazy-discovery | 0.0% | 0.0% | 0.0% | 0.0% |

So the treatment changed how confusable a twin is *once seen*, not how often it is seen —
which is exactly what changing one sentence of description ought to do, and is why the
comparison is interpretable at all.

The mechanism is that a generic clause is legible as filler. *"Results are not
paginated"* attaches to any tool and therefore discriminates none, so a model can
discount it and fall back on the name. A domain-specific clause reads as real content, so
the synthetic stops looking like a decoy and starts looking like a genuine sibling —
`git_diff_unstaged` beside a tool that also talks about worktrees and staged hunks. v2's
distractors are better distractors precisely because they are more plausible, and
plausibility is indistinguishable from correctness when the label blesses exactly one of
the pair.

`static-hot-set` is the one arm essentially unaffected (+1.0, p=0.82). It is also the
only arm that promotes real, frequency-ranked tools with full schemas while never
promoting a twin schema — 27.9% target exposure against 0.0% twin exposure. A plausible
reading is that a promoted correct schema crowds out the twin regardless of how good the
twin's description got. With six arms and two corpus versions that is a hypothesis worth
testing, not a result.

## What this means for reading any accuracy number here

Strict accuracy is not a property of the router. It is a property of the router **and**
the distractor generator, and the generator moved it by up to 9 points between two
versions that differ by one sentence:

| arm | v1 strict | v2 strict | delta | v1 lenient | v2 lenient | delta |
|---|---|---|---|---|---|---|
| **hotset** | 42.6% | 31.0% | **−11.6** | 62.9% | 61.9% | −1.0 |
| **index-only** | 40.6% | 31.6% | **−9.0** | 56.5% | 54.5% | −1.9 |
| rag-over-tools | 34.2% | 33.2% | −1.0 | 49.7% | 52.3% | +2.6 |
| lazy-discovery | 57.1% | 55.8% | −1.3 | 60.0% | 61.9% | +1.9 |
| static-hot-set | 40.3% | 43.5% | +3.2 | 53.9% | 58.1% | +4.2 |
| full-catalog | 33.2% | 36.5% | +3.2 | 43.5% | 57.7% | +14.2 |

`hotset` loses **11.6 points of strict accuracy** between v1 and v2 while its lenient
accuracy moves 1.0. Nothing about the arm changed: the hot set is empty on qwen-flash in
both runs, and the plan replay above confirms its exposure is the same to within a point.
Under v1 it is the second-best strict arm in the table; under v2 it is second-worst.
`index-only` does the same thing at −9.0 strict against −1.9 lenient.

Any ranking that quotes a single strict column is reporting the corpus as much as the
router. The direction is not even stable across arms — `full-catalog` and
`static-hot-set` gained strict accuracy over the same change that cost `hotset` 11.6
points, so this cannot be corrected with an offset.

This is also why lenient is not a charitable rounding. Strict and lenient bracket a real
quantity — strict cannot tell a wrong tool from an equivalent one, lenient cannot tell a
right tool from a lucky one — and the width of that bracket is set by how aggressive the
distractor generator was. Reporting both, with `twin_of` provenance, is the only way the
number survives a change to the corpus.

## Caveats

- Two corpus versions and six arms. The per-arm deltas above are measurements; any
  account of *why* one arm moved 11.6 points and another moved 1.0 is a hypothesis this
  design cannot separate — exposure, promotion of the correct schema, and retrieval
  behaviour all covary across these arms.
- An earlier draft of this file claimed the twin-rate increase was monotone in how much
  catalog an arm exposes. The plan replay refutes that: `index-only` shows no twin
  schemas at all and still moved +7.1 points.
- Twin provenance covers synthetic distractors only. Genuine near-duplicates between two
  real MCP servers are scored strictly in both versions, so lenient is a lower bound in
  both and the v1-v2 delta is not affected by them.
- The clause pools differ in size and wording, so "generic vs domain-specific" is the
  intended contrast but not a perfectly isolated one.
