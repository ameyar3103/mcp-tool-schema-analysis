"""Is admission losing money because the price is wrong, or because the forecast is?

Replays the controller offline and, for every tool it admits, compares the uses the
predictor promised against the uses the trace actually delivered over the same window.
No API calls: the decision is a pure function of the trace.
"""

from __future__ import annotations

import sys

from hotset.config import MODELS
from hotset.corpus.distractors import pad_catalog
from hotset.corpus.harvest import load
from hotset.eval.tasks import load as load_tasks
from hotset.eval.tasks import split
from hotset.eval.workload import concentration, repeat
from hotset.policy.adaptive import HotSet
from hotset.policy.economics import asymptotic_rate
from hotset.policy.predictors import LRUK, Oracle, warm

HORIZON = 50


def admissions(arm: HotSet, catalog, sessions, flat: list[str]) -> list[tuple[float, int]]:
    """(predicted, actual) uses per distinct admission, scored at the turn it was admitted."""
    seen: set[str] = set()
    rows: list[tuple[float, int]] = []
    turn = 0
    for session in sessions:
        arm.reset()
        for t in session.turns:
            arm.plan(catalog, [], t.user)
            for tool in arm.hot:
                if tool.name not in seen:
                    seen.add(tool.name)
                    # Same window the prediction covered, so the two are comparable.
                    actual = flat[turn : turn + HORIZON].count(tool.name)
                    rows.append((arm.predictor.expected_uses(tool.name, HORIZON), actual))
            turn += 1
            arm.observe(t.tool)
    return rows


def report(
    model: str, rate: float, predictor: str, rows: list[tuple[float, int]], bar: float, peak: int
) -> None:
    """One line per configuration. `earned` is the only column that decides anything."""
    predicted = sum(p for p, _ in rows)
    actual = sum(a for _, a in rows)
    earned = sum(1 for _, a in rows if a >= bar)
    print(
        f"{model:11} reuse={rate:<4} peak/{HORIZON}={peak:<3} {predictor:6} "
        f"admitted={len(rows):3} predicted={predicted:7.1f} actual={actual:4} "
        f"over={predicted / max(actual, 1):5.1f}x earned={earned}/{len(rows)}"
    )


def main(model: str = "haiku", size: int = 300, version: int = 2) -> None:
    spec = MODELS[model]
    base = load()
    catalog = base if size == len(base) else pad_catalog(base, size, seed=0, version=version)
    train, test = split(load_tasks())
    # The floor threshold, reached only when the hot set is empty; admitting raises it.
    # Scoring against the floor makes `earned` an upper bound on admissions that paid.
    bar = asymptotic_rate(spec) * HORIZON

    for rate in (0.0, 1.0, 3.0):
        sessions = repeat(test, rate=rate, seed=0) if rate else test
        flat = [t.tool for s in sessions for t in s.turns]
        peak = concentration(sessions)[1]
        for name in ("lru-k", "oracle"):
            if name == "lru-k":
                predictor = LRUK(k=2)
                warm(predictor, ([t.tool for t in s.turns] for s in train))
            else:
                predictor = Oracle(flat)  # upper bound: separates weak forecast from bad price
            arm = HotSet(spec, predictor, horizon=HORIZON)
            report(model, rate, name, admissions(arm, catalog, sessions, flat), bar, peak)


USAGE = "usage: calibration.py [model] [size] [version]"

if __name__ == "__main__":
    casts = [str, int, int]
    if len(sys.argv) - 1 > len(casts):
        raise SystemExit(USAGE)
    names = ["model", "size", "version"]
    main(**{n: c(v) for n, c, v in zip(names, casts, sys.argv[1:])})
