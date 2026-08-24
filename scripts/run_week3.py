"""Full sweep: baselines, the index-only ablation, and cache-aware admission."""

from __future__ import annotations

import sys
import uuid

from hotset.config import MODELS
from hotset.corpus.distractors import pad_catalog
from hotset.corpus.harvest import load
from hotset.eval.runner import run_arm, save, summarize
from hotset.eval.spans import Recorder
from hotset.eval.tasks import load as load_tasks
from hotset.eval.tasks import split
from hotset.eval.workload import concentration, trace
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

HEAD = (
    f"{'arm':16} {'acc':>6} {'twin':>6} {'lenient':>8} {'halluc':>7} {'hit':>6} "
    f"{'prompt':>8} {'hops':>5} {'$/turn':>10} {'$/correct':>10}"
)
# One shared prefix serves all traffic, so admission amortizes over the deployment,
# not over a single five-turn conversation.
HORIZON = 50


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
    salt = uuid.uuid4().hex[:8]

    predictor = LRUK(k=2)
    warm(predictor, ([t.tool for t in s.turns] for s in train))

    arms = [
        FullCatalog(),
        RagOverTools(k=8),
        LazyDiscovery(),
        IndexOnly(),
        StaticHotSet(frequency_hot_set(catalog, train, 16)),
        HotSet(spec, predictor, horizon=HORIZON),
    ]
    print(
        f"{model} | catalog {len(catalog)} | test {len(test)} sessions "
        f"({sum(len(s.turns) for s in test)} turns) | skew {skew} "
        f"| top5 {concentration(test)[0]:.1%} peak/50 {concentration(test)[1]} | salt {salt}\n\n{HEAD}"
    )
    for arm in arms:
        tag = f"{arm.name}-{model}-{len(catalog)}v{version}s{skew:g}-{salt}"
        rec = Recorder(arm.name, spec.slug)  # traces answer "did it degrade mid-run"
        results = run_arm(arm, spec, catalog, test, salt=salt, recorder=rec)
        save(results, tag)
        rec.save(tag)
        m = summarize(results)
        print(
            f"{arm.name:16} {m['accuracy']:6.1%} {m['twin']:6.1%} {m['lenient_accuracy']:8.1%} "
            f"{m['hallucinated']:7.1%} {m['hit_rate']:6.1%} "
            f"{m['prompt_tokens']:8.0f} {m['hops']:5.2f} "
            f"${m['cost_per_turn']:.6f} ${m['cost_per_correct']:.6f}"
            + (f"  ({m['errors']} err)" if m["errors"] else "")
        )
    if isinstance(arms[-1], HotSet):
        print(f"\nhot set at end ({len(arms[-1].hot)}): {[t.name for t in arms[-1].hot]}")


if __name__ == "__main__":
    opts = {}
    if len(sys.argv) > 2:
        opts["size"] = int(sys.argv[2])
    if len(sys.argv) > 3:
        opts["version"] = int(sys.argv[3])
    if len(sys.argv) > 4:
        opts["skew"] = float(sys.argv[4])
    main(*(sys.argv[1:2] or []), **opts)
