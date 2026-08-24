"""Replay the eval trace through the policy without calling the API.

Admission is a function of token counts, prices and the predictor, none of which need
a model response. Simulating first says which predictors would actually differ, so the
paid sweep only runs arms whose prompts are not byte-identical.
"""

from __future__ import annotations

import sys

from hotset.config import MODELS
from hotset.corpus.distractors import pad_catalog
from hotset.corpus.harvest import load
from hotset.eval.tasks import load as load_tasks
from hotset.eval.tasks import split
from hotset.eval.workload import concentration, trace
from hotset.layout.tokens import exact
from hotset.policy.adaptive import HotSet
from hotset.policy.economics import rewritten_segment
from hotset.policy.predictors import LRUK, Ensemble, Markov, Oracle, warm

HORIZON = 50


def replay(policy: HotSet, catalog, sessions) -> tuple[int, int, float, list[str]]:
    """Peak hot-set size, admissions, the dollars those admissions cost, and membership.

    Every membership change rewrites the layer B segment, so the write is billed again.
    That cost is knowable without calling the model, which is the point: a predictor
    that over-admits can be priced before a single token is spent.
    """
    peak, admissions, spend, previous = 0, 0, 0.0, set()
    for session in sessions:
        policy.reset()
        for turn in session.turns:
            policy.plan(catalog, [], turn.user)
            current = {t.name for t in policy.hot}
            if current != previous:
                hot_tokens = sum(policy._schema(t) for t in policy.hot)
                segment = rewritten_segment(policy.spec, policy._head(catalog), hot_tokens)
                spend += segment * policy.spec.write / 1e6
            admissions += len(current - previous)
            peak, previous = max(peak, len(current)), current
            policy.observe(turn.tool)  # the deployment only sees what was called
    return peak, admissions, spend, sorted(previous)


def main(
    model: str = "qwen-flash", size: int = 300, version: int = 2, skew: float = 0.0
) -> None:
    spec = MODELS[model]
    base = load()
    catalog = base if size == len(base) else pad_catalog(base, size, seed=0, version=version)
    train, test = split(load_tasks())
    if skew:
        # Same questions, different arrival mix: isolates workload concentration from
        # task difficulty, which a freshly generated skewed suite could not.
        servers = {t.name: t.server for t in base}
        test = trace(test, servers, skew=skew, length=len(test), seed=0)
    future = [t.tool for s in test for t in s.turns]
    global HORIZON
    HORIZON = len(future)  # price against the horizon actually measured, not a guess

    top5, peak = concentration(test)
    print(
        f"{model} | catalog {len(catalog)} v{version} | {len(future)} turns | skew {skew} "
        f"| top5 {top5:.1%} peak/50 {peak} | horizon {HORIZON}"
        # Admission is a token-count decision, so which counter ran is part of the result.
        f" | tokens {'reference BPE' if exact() else 'CHAR FALLBACK'}\n"
    )
    print(f"{'predictor':12} {'peak':>5} {'admits':>7} {'rewrite $':>10}  final")
    for name, predictor in [
        ("lru-k", LRUK(k=2)),
        ("markov", Markov()),
        ("ensemble", Ensemble([LRUK(k=2), Markov()])),
        ("oracle", Oracle(future)),
    ]:
        if name != "oracle":
            warm(predictor, ([t.tool for t in s.turns] for s in train))
        peak, admits, spend, final = replay(
            HotSet(spec, predictor, horizon=HORIZON), catalog, test
        )
        print(f"{name:12} {peak:5} {admits:7} {spend:10.4f}  {final}")


if __name__ == "__main__":
    opts = {}
    if len(sys.argv) > 2:
        opts["size"] = int(sys.argv[2])
    if len(sys.argv) > 3:
        opts["version"] = int(sys.argv[3])
    if len(sys.argv) > 4:
        opts["skew"] = float(sys.argv[4])
    main(*(sys.argv[1:2] or []), **opts)
