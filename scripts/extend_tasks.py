"""Grow the frozen task suite until it can resolve the gaps week 4 needs to measure.

At 195 turns the suite's minimum detectable accuracy gap is 7.2%, which is wider than
any predictor ablation is expected to move. Power under McNemar scales with discordant
pairs, so the only lever is more turns.
"""

from __future__ import annotations

import random

from hotset.corpus.harvest import load
from hotset.eval.significance import minimum_detectable
from hotset.eval.tasks import freeze, generate
from hotset.eval.tasks import load as load_tasks

TARGET, PER_BATCH, TURNS, WINDOW = 120, 5, 5, 30


def main() -> None:
    catalog = load()
    sessions = load_tasks()
    seen = {s.scenario for s in sessions}
    rng = random.Random(0)
    batch = 0
    while len(sessions) < TARGET and batch < 40:
        batch += 1
        # A rotating slice, not the whole catalog: the same index every time yields the
        # same scenarios, and coverage stays stuck on the tools that read most obviously.
        focus = rng.sample(catalog, min(WINDOW, len(catalog)))
        try:
            fresh = [
                s
                for s in generate(focus, n=PER_BATCH, turns=TURNS, avoid=seen)
                if s.scenario not in seen
            ]
        except Exception as exc:  # noqa: BLE001 - one bad batch must not lose the rest
            print(f"batch {batch}: failed {type(exc).__name__}: {exc}")
            continue
        seen.update(s.scenario for s in fresh)
        sessions.extend(fresh)
        freeze(sessions)  # checkpoint every batch; generation is the expensive part
        print(f"batch {batch}: +{len(fresh)} -> {len(sessions)} sessions")
    turns = [t for s in sessions for t in s.turns]
    held_out = round(len(turns) * 0.5)
    print(
        f"\ntotal {len(sessions)} sessions / {len(turns)} turns / "
        f"{len({t.tool for t in turns})} distinct tools\n"
        f"held-out {held_out} turns -> minimum detectable gap {minimum_detectable(held_out):.1%}"
    )


if __name__ == "__main__":
    main()
