"""Offline: how often does each arm actually show the target's twin, or the target?

A twin-selection rate answers "how often did the arm pick a near-duplicate", but not
whether it could have. Replaying every plan without calling a model separates the two:
if twin confusion rises between corpus versions while exposure holds flat, the corpus
changed how confusable a twin is, not how often it is seen.
"""

from __future__ import annotations

import sys

from hotset.config import MODELS
from hotset.corpus.distractors import pad_catalog
from hotset.corpus.harvest import load
from hotset.eval.tasks import load as load_tasks
from hotset.eval.tasks import split
from hotset.policy.adaptive import HotSet
from hotset.policy.baselines import (
    FullCatalog,
    IndexOnly,
    LazyDiscovery,
    RagOverTools,
    StaticHotSet,
    frequency_hot_set,
)
from hotset.policy.predictors import LRUK, warm

HORIZON = 50


def arms(catalog, train, spec):
    """The week-3 sweep, rebuilt identically so the replay matches the paid run."""
    predictor = LRUK(k=2)
    warm(predictor, ([t.tool for t in s.turns] for s in train))
    return [
        FullCatalog(),
        RagOverTools(k=8),
        LazyDiscovery(),
        IndexOnly(),
        StaticHotSet(frequency_hot_set(catalog, train, 16)),
        HotSet(spec, predictor, horizon=HORIZON),
    ]


def exposure(arm, catalog, sessions, twin_of) -> dict:
    """Per-arm counts over turns whose labeled tool has a synthetic twin at all."""
    seen = {"hot": 0, "tail": 0, "either": 0, "target": 0, "n": 0}
    for session in sessions:
        history: list[dict] = []
        for turn in session.turns:
            twin = twin_of.get(turn.tool)
            if twin:
                seen["n"] += 1
                plan = arm.plan(catalog, history, turn.user)
                in_hot = any(t.name == twin for t in plan.hot)
                in_tail = any(t.name == twin for t in plan.tail)
                seen["hot"] += in_hot
                seen["tail"] += in_tail
                seen["either"] += in_hot or in_tail
                # The target's own schema: a promoted correct answer may crowd out a twin.
                seen["target"] += any(t.name == turn.tool for t in plan.hot + plan.tail)
            history.append({"role": "user", "content": turn.user})
    return seen


def main(model: str = "qwen-flash", size: int = 300, version: int = 2) -> None:
    size, version = int(size), int(version)  # argv arrives as strings
    spec = MODELS[model]
    base = load()
    catalog = pad_catalog(base, size, seed=0, version=version)
    train, test = split(load_tasks())
    twin_of = {t.twin_of: t.name for t in catalog if t.twin_of}

    print(f"{model} | catalog {size} | corpus v{version}")
    print(f"{'arm':16} {'twin in hot':>12} {'twin in tail':>13} {'target shown':>13}   n")
    for arm in arms(catalog, train, spec):
        s = exposure(arm, catalog, test, twin_of)
        n = s["n"] or 1
        print(
            f"{arm.name:16} {s['hot'] / n:11.1%} {s['tail'] / n:12.1%} "
            f"{s['target'] / n:12.1%}   {s['n']}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
