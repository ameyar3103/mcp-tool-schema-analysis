"""Synthetic near-duplicate distractors that pad the catalog for scale stress tests.

Random junk tools would be trivially separable. The documented failure is name
collision between plausible neighbours, so distractors have to be confusable.
"""

from __future__ import annotations

import random

from hotset.corpus.models import Tool


def make_distractor(source: Tool, rng: random.Random) -> Tool:
    """Clone a real tool under a confusable but still distinguishable identity."""
    # TODO(human)
    raise NotImplementedError


def pad_catalog(tools: list[Tool], target: int, seed: int = 0) -> list[Tool]:
    """Grow the catalog to `target` tools; seeded so the corpus stays reproducible."""
    rng = random.Random(seed)
    out = list(tools)
    seen = {t.name for t in tools}
    for _ in range(target * 100):  # bounded: name space may be smaller than target
        if len(out) >= target:
            break
        candidate = make_distractor(rng.choice(tools), rng)
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        out.append(candidate)
    return out
