"""Predictor ablation: does knowing *what follows what* change what gets admitted?

Every arm is the same HotSet policy with the same economics; only the predictor
differs. Any accuracy or cost gap is therefore attributable to prediction alone,
which the week 3 sweep could not isolate.
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
from hotset.eval.workload import concentration, repeat, trace
from hotset.policy.adaptive import HotSet
from hotset.policy.baselines import IndexOnly
from hotset.policy.predictors import LRUK, Ensemble, Markov, Oracle, warm

HEAD = (
    f"{'arm':16} {'acc':>6} {'lenient':>8} {'hit':>6} {'prompt':>8} {'hot':>4} "
    f"{'$/turn':>10} {'$/correct':>10}"
)
HORIZON = 50


def build(spec, train, test):
    """Predictors warmed on train only. Oracle is the exception, by definition."""
    future = [t.tool for s in test for t in s.turns]
    made = []
    for name, predictor in [
        ("lru-k", LRUK(k=2)),
        ("markov", Markov()),
        ("ensemble", Ensemble([LRUK(k=2), Markov()])),
    ]:
        warm(predictor, ([t.tool for t in s.turns] for s in train))
        arm = HotSet(spec, predictor, horizon=HORIZON)
        arm.name = f"hotset-{name}"
        made.append(arm)
    oracle = HotSet(spec, Oracle(future), horizon=HORIZON)
    oracle.name = "hotset-oracle"
    made.append(oracle)
    return made


def main(
    model: str = "qwen-flash",
    size: int = 300,
    version: int = 2,
    skew: float = 0.0,
    reuse: float = 0.0,
) -> None:
    spec = MODELS[model]
    base = load()
    catalog = base if size == len(base) else pad_catalog(base, size, seed=0, version=version)
    train, test = split(load_tasks())
    if reuse:
        # Applied before build(), so the oracle's future is the trace actually served.
        # Without reuse the oracle admits nothing and every arm emits identical plans.
        test = repeat(test, rate=reuse, seed=0)
    if skew:
        # Same questions, different arrival mix: isolates workload concentration from
        # task difficulty, which a freshly generated skewed suite could not.
        servers = {t.name: t.server for t in base}
        test = trace(test, servers, skew=skew, length=len(test), seed=0)
    salt = uuid.uuid4().hex[:8]

    arms = [IndexOnly(), *build(spec, train, test)]  # index-only is the no-admission floor
    print(
        f"{model} | catalog {len(catalog)} | test {len(test)} sessions "
        f"({sum(len(s.turns) for s in test)} turns) | skew {skew} reuse {reuse} "
        f"| top5 {concentration(test)[0]:.1%} peak/50 {concentration(test)[1]} | salt {salt}\n\n{HEAD}"
    )
    collected = {}
    for arm in arms:
        tag = f"{arm.name}-{model}-{len(catalog)}v{version}s{skew:g}r{reuse:g}-{salt}"
        rec = Recorder(arm.name, spec.slug)  # traces answer "did it degrade mid-run"
        results = run_arm(arm, spec, catalog, test, salt=salt, recorder=rec)
        save(results, tag)
        rec.save(tag)
        collected[arm.name] = results
        m = summarize(results)
        hot = len(arm.hot) if isinstance(arm, HotSet) else 0
        print(
            f"{arm.name:16} {m['accuracy']:6.1%} {m['lenient_accuracy']:8.1%} {m['hit_rate']:6.1%} "
            f"{m['prompt_tokens']:8.0f} {hot:4} "
            f"${m['cost_per_turn']:.6f} ${m['cost_per_correct']:.6f}"
            + (f"  ({m['errors']} err)" if m["errors"] else "")
        )
    # Both views: on a catalog padded with near-duplicates, a strict-only comparison
    # measures which twin the label picked as much as it measures the predictor.
    for lenient in (False, True):
        print(f"\n{'lenient (twin counts as correct)' if lenient else 'strict'}")
        for c in compare(collected, lenient):
            mark = "  <-- significant" if c.significant else ""
            print(f"{c.a:16} vs {c.b:16} {c.a_only:3}/{c.b_only:<3}  p={c.p_value:.3f}{mark}")


USAGE = "usage: run_week4.py [model] [size] [version] [skew] [reuse]"

if __name__ == "__main__":
    # Typed positionals, extras rejected: a silently ignored argument once bought a
    # full paid sweep of the workload it was meant to replace.
    casts = [str, int, int, float, float]
    if len(sys.argv) - 1 > len(casts):
        raise SystemExit(USAGE)
    names = ["model", "size", "version", "skew", "reuse"]
    main(**{n: c(v) for n, c, v in zip(names, casts, sys.argv[1:])})
