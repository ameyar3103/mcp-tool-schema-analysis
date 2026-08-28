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
    """The paid sweep's arms, rebuilt identically so the replay matches it."""
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
    twins = {t.name for t in catalog if t.twin_of}
    seen = {"hot": 0, "tail": 0, "either": 0, "target": 0, "any": 0, "n": 0}
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
                shown = plan.hot + plan.tail
                seen["target"] += any(t.name == turn.tool for t in shown)
                # Any twin at all, not just the target's: the distractor-amplification rate.
                seen["any"] += any(t.name in twins for t in shown)
            history.append({"role": "user", "content": turn.user})
    return seen


def main(model: str = "haiku", size: int = 300, version: int = 2) -> None:
    spec = MODELS[model]
    base = load()
    catalog = pad_catalog(base, size, seed=0, version=version)
    train, test = split(load_tasks())
    twin_of = {t.twin_of: t.name for t in catalog if t.twin_of}

    print(f"{model} | catalog {size} | corpus v{version}")
    print(
        f"{'arm':16} {'twin in hot':>12} {'twin in tail':>13} "
        f"{'target shown':>13} {'any twin':>9}   n"
    )
    for arm in arms(catalog, train, spec):
        s = exposure(arm, catalog, test, twin_of)
        n = s["n"] or 1
        print(
            f"{arm.name:16} {s['hot'] / n:11.1%} {s['tail'] / n:12.1%} "
            f"{s['target'] / n:12.1%} {s['any'] / n:8.1%}   {s['n']}"
        )


USAGE = "usage: twin_exposure.py [model] [size] [version]"

if __name__ == "__main__":
    casts = [str, int, int]
    if len(sys.argv) - 1 > len(casts):
        raise SystemExit(USAGE)
    names = ["model", "size", "version"]
    main(**{n: c(v) for n, c, v in zip(names, casts, sys.argv[1:])})
