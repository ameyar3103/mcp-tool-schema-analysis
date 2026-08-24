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
from hotset.eval.tasks import load as load_tasks
from hotset.eval.tasks import split
from hotset.policy.adaptive import HotSet
from hotset.policy.baselines import IndexOnly
from hotset.policy.predictors import LRUK, Ensemble, Markov, Oracle, warm

HEAD = f"{'arm':16} {'acc':>6} {'hit':>6} {'prompt':>8} {'hot':>4} {'$/turn':>10} {'$/correct':>10}"
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


def main(model: str = "qwen-flash", size: int = 300) -> None:
    spec = MODELS[model]
    base = load()
    catalog = base if size == len(base) else pad_catalog(base, size, seed=0)
    train, test = split(load_tasks())
    salt = uuid.uuid4().hex[:8]

    arms = [IndexOnly(), *build(spec, train, test)]  # index-only is the no-admission floor
    print(
        f"{model} | catalog {len(catalog)} | test {len(test)} sessions "
        f"({sum(len(s.turns) for s in test)} turns) | salt {salt}\n\n{HEAD}"
    )
    collected = {}
    for arm in arms:
        results = run_arm(arm, spec, catalog, test, salt=salt)
        save(results, f"{arm.name}-{model}-{len(catalog)}-{salt}")
        collected[arm.name] = results
        m = summarize(results)
        hot = len(arm.hot) if isinstance(arm, HotSet) else 0
        print(
            f"{arm.name:16} {m['accuracy']:6.1%} {m['hit_rate']:6.1%} "
            f"{m['prompt_tokens']:8.0f} {hot:4} "
            f"${m['cost_per_turn']:.6f} ${m['cost_per_correct']:.6f}"
            + (f"  ({m['errors']} err)" if m["errors"] else "")
        )
    print()
    for c in compare(collected):
        mark = "  <-- significant" if c.significant else ""
        print(f"{c.a:16} vs {c.b:16} {c.a_only:3}/{c.b_only:<3}  p={c.p_value:.3f}{mark}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []), **({"size": int(sys.argv[2])} if len(sys.argv) > 2 else {}))
