"""Synthetic near-duplicate distractors that pad the catalog for scale stress tests.

Random junk tools would be trivially separable. The documented failure is name
collision between plausible neighbours, so distractors have to be confusable.
"""

from __future__ import annotations

import random

from hotset.corpus.models import Tool

# Verb synonyms: the primary collision axis, applied to names and descriptions alike.
_SYNONYMS = [
    ("get", "fetch"), ("fetch", "retrieve"), ("create", "make"), ("make", "add"),
    ("delete", "remove"), ("remove", "drop"), ("update", "modify"), ("modify", "edit"),
    ("list", "enumerate"), ("read", "load"), ("write", "save"), ("search", "find"),
    ("open", "launch"), ("close", "dismiss"), ("navigate", "goto"), ("select", "choose"),
]
# Fallbacks so tools without a swappable verb can still spawn variants.
_NAMESPACES = ["ext", "alt", "aux", "legacy", "beta"]
_QUALIFIERS = ["v2", "ex", "raw", "sync", "strict", "batch"]
# Plausible differentiators: enough divergence that a unique right answer survives.
_CLAUSES = [
    "Results are not paginated.",
    "Paths resolve relative to the workspace root.",
    "Fails instead of overwriting existing entries.",
    "Includes hidden and ignored entries.",
    "Returns a compact representation without metadata.",
    "Runs without acquiring a lock.",
]


def _swap_synonym(text: str, rng: random.Random) -> str | None:
    """Swap one verb for a plausible synonym; None when no synonym applies."""
    options = [(a, b) for a, b in _SYNONYMS if a in text]
    if not options:
        return None
    old, new = rng.choice(options)
    return text.replace(old, new, 1)


def _variant_name(source: Tool, rng: random.Random) -> str:
    """Confusable name, via synonym swap where possible and namespacing otherwise."""
    swapped = _swap_synonym(source.name, rng)
    if swapped and rng.random() < 0.5:
        return swapped
    base = swapped or source.name
    style = rng.choice(("suffix", "prefix", "reorder"))
    if style == "suffix":
        return f"{base}_{rng.choice(_QUALIFIERS)}"
    if style == "prefix":
        return f"{rng.choice(_NAMESPACES)}_{base}"
    parts = base.split("_")
    rng.shuffle(parts)
    return "_".join(parts)


def make_distractor(source: Tool, rng: random.Random) -> Tool:
    """Clone a real tool under a confusable but still distinguishable identity."""
    # Description keeps the source's shape: a giveaway marker would make synthetics
    # trivially separable and collapse the token accounting the benchmark depends on.
    desc = _swap_synonym(source.description, rng) or source.description
    return Tool(
        name=_variant_name(source, rng),
        description=f"{desc.rstrip().rstrip('.')}. {rng.choice(_CLAUSES)}",
        input_schema=source.input_schema,  # unchanged: keeps token accounting realistic
        server=f"{source.server}-alt",
        synthetic=True,
    )


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
