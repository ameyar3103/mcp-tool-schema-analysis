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

## v2 made twins harder, and the effect scales with catalog exposure

v2 was written to make synthetics *more* distinguishable. It did the opposite, and the
size of the reversal is ordered by how much of the catalog each arm puts in front of the
model:

| arm | what it exposes | v1 twin | v2 twin | delta | Fisher p |
|---|---|---|---|---|---|
| static-hot-set | hot block | 13.5% | 14.5% | +1.0 | 0.8172 |
| lazy-discovery | ~0 names | 2.9% | 6.1% | +3.2 | 0.0801 |
| rag-over-tools | k retrieved | 15.5% | 19.0% | +3.5 | 0.2879 |
| index-only | 300 names | 15.8% | 22.9% | **+7.1** | **0.0325** |
| full-catalog | 300 names + schemas | 10.3% | 21.3% | **+11.0** | **0.0003** |

Monotone in exposure, and the only two arms that move significantly are the two that
hold the entire catalog. That is the dose-response curve the twin-confusion story
predicts: an arm can only be confused by a near-duplicate it can actually see.

The mechanism is that a generic clause is legible as filler. *"Results are not
paginated"* attaches to any tool and therefore discriminates none, so a model can
discount it and fall back on the name. A domain-specific clause reads as real content, so
the synthetic stops looking like a decoy and starts looking like a genuine sibling —
`git_diff_unstaged` next to a tool that also talks about worktrees and staged hunks. v2's
distractors are better distractors precisely because they are more plausible, and
plausibility is indistinguishable from correctness when the label blesses exactly one of
the pair.

## What this means for reading any accuracy number here

Strict accuracy is not a property of the router. It is a property of the router **and**
the distractor generator, and the generator moved it by up to 9 points between two
versions that differ by one sentence:

| arm | v1 strict | v2 strict | v1 lenient | v2 lenient |
|---|---|---|---|---|
| lazy-discovery | 57.1% | 55.8% | 60.0% | 61.9% |
| static-hot-set | 40.3% | 43.5% | 53.9% | 58.1% |
| full-catalog | 33.2% | 36.5% | 43.5% | 57.7% |
| index-only | 40.6% | 31.6% | 56.5% | 54.5% |
| rag-over-tools | 34.2% | 33.2% | 49.7% | 52.3% |

`index-only` loses 9.0 points of strict accuracy going from v1 to v2 while its lenient
accuracy moves 2.0. Nothing about the arm changed. Under v1 it is the second-best strict
arm; under v2 it is the worst. Any ranking that quotes a single strict column is
reporting the corpus as much as the router.

This is also why lenient is not a charitable rounding. Strict and lenient bracket a real
quantity — strict cannot tell a wrong tool from an equivalent one, lenient cannot tell a
right tool from a lucky one — and the width of that bracket is set by how aggressive the
distractor generator was. Reporting both, with `twin_of` provenance, is the only way the
number survives a change to the corpus.

## Caveats

- Two corpus versions is two points. The monotonicity above is suggestive of a
  dose-response, not a measured slope.
- Twin provenance covers synthetic distractors only. Genuine near-duplicates between two
  real MCP servers are scored strictly in both versions, so lenient is a lower bound in
  both and the v1-v2 delta is not affected by them.
- The clause pools differ in size and wording, so "generic vs domain-specific" is the
  intended contrast but not a perfectly isolated one.
