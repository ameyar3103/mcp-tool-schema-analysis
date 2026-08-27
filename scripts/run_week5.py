"""How big should the cached hot set be? The schema-budget frontier.

Week 3 asked whether a tool clears break-even. That is the wrong selector, for two
reasons the offline replay makes plain:

  - Break-even prices tokens under the assumption that a tail-load substitutes for a
    cached schema. It does not. `static-hot-set` carries 16 schemas the threshold would
    reject and still beats `index-only` by 5.2 points strict, so a schema buys accuracy
    the cost model has no term for.
  - LRU-K adaptivity is a liability on stationary traffic. At a matched budget its hot
    set covers the called tool less often than a plain train-frequency ranking does
    (20.3% vs 27.7% at K=16; 75.8% vs 90.6% at K=64).

So the controlled variable here is the *budget*, not the threshold: sweep K from 0
(`index-only`) to 300 (`full-catalog`) with membership fixed for the whole run, and put
the adaptive arm at the best budget to test whether adaptivity earns anything at all.
Train-frequency coverage of the held-out calls is 14.8/27.7/55.2/90.6% at K=8/16/32/64.
"""

from __future__ import annotations

import sys
import uuid

from hotset.config import MODELS
from hotset.corpus.distractors import pad_catalog
from hotset.corpus.harvest import load
from hotset.eval.runner import run_arm, save, summarize
from hotset.eval.significance import compare
from hotset.eval.spans import Recorder
from hotset.eval.tasks import load as load_tasks
from hotset.eval.tasks import split
from hotset.eval.workload import concentration, repeat
from hotset.policy.adaptive import HotSet
from hotset.policy.baselines import FullCatalog, IndexOnly, StaticHotSet, frequency_hot_set
from hotset.policy.predictors import LRUK, warm

HEAD = (
    f"{'arm':22} {'strict':>7} {'twin':>6} {'lenient':>8} {'hit':>6} {'prompt':>8} "
    f"{'hot':>4} {'rw':>4} {'$/turn':>10} {'$/correct':>10}"
)
HORIZON = 50


def main(
    model: str = "haiku",
    size: int = 300,
    version: int = 2,
    reuse: float = 0.0,
) -> None:
    spec = MODELS[model]
    base = load()
    catalog = base if size == len(base) else pad_catalog(base, size, seed=0, version=version)
    train, test = split(load_tasks())
    if reuse:
        test = repeat(test, rate=reuse, seed=0)
    salt = uuid.uuid4().hex[:8]

    def static(k: int) -> StaticHotSet:
        arm = StaticHotSet(frequency_hot_set(catalog, train, k))
        arm.name = f"static-{k}"
        return arm

    def adaptive(budget: int) -> HotSet:
        """Matched budget, so any gap is the predictor and not the token spend."""
        predictor = LRUK(k=2)
        warm(predictor, ([t.tool for t in s.turns] for s in train))
        # tail_k=0: BM25 top-3 puts a twin's full schema in the prompt on 98% of turns
        # and the target's on 37%, so on a twin-padded catalog layer C feeds the impostor.
        arm = HotSet(spec, predictor, horizon=HORIZON, epoch=HORIZON, tail_k=0, budget=budget)
        arm.name = f"adaptive-{budget}"
        return arm

    arms = [
        IndexOnly(),  # K=0, the floor
        static(16),
        static(32),
        static(64),
        adaptive(64),  # same budget as static-64; isolates prediction from spend
        FullCatalog(),  # K=300, the ceiling
    ]
    print(
        f"{model} | catalog {len(catalog)} | test {len(test)} sessions "
        f"({sum(len(s.turns) for s in test)} turns) | reuse {reuse} "
        f"| peak/50 {concentration(test)[1]} | salt {salt}\n\n{HEAD}"
    )
    collected = {}
    for arm in arms:
        tag = f"{arm.name}-{model}-{len(catalog)}v{version}r{reuse:g}-{salt}"
        rec = Recorder(arm.name, spec.slug)
        results = run_arm(arm, spec, catalog, test, salt=salt, recorder=rec)
        save(results, tag)
        rec.save(tag)
        collected[arm.name] = results
        m = summarize(results)
        hot = len(arm.hot) if isinstance(arm, HotSet) else 0
        print(
            f"{arm.name:22} {m['accuracy']:7.1%} {m['twin']:6.1%} {m['lenient_accuracy']:8.1%} "
            f"{m['hit_rate']:6.1%} {m['prompt_tokens']:8.0f} {hot:4} "
            f"{getattr(arm, 'rewrites', 0):4} "
            f"${m['cost_per_turn']:.6f} ${m['cost_per_correct']:.6f}"
            + (f"  ({m['errors']} err)" if m["errors"] else "")
        )
    for lenient in (False, True):
        print(f"\n{'lenient (twin counts as correct)' if lenient else 'strict'}")
        for c in compare(collected, lenient):
            mark = "  <-- significant" if c.significant else ""
            print(f"{c.a:22} vs {c.b:22} {c.a_only:3}/{c.b_only:<3} p={c.p_value:.3f}{mark}")


USAGE = "usage: run_week5.py [model] [size] [version] [reuse]"

if __name__ == "__main__":
    casts = [str, int, int, float]
    if len(sys.argv) - 1 > len(casts):
        raise SystemExit(USAGE)
    names = ["model", "size", "version", "reuse"]
    main(**{n: c(v) for n, c, v in zip(names, casts, sys.argv[1:])})
