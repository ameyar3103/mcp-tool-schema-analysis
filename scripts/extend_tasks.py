"""Grow the frozen task suite. Existing scenarios are kept; duplicates are dropped."""

from __future__ import annotations

from hotset.corpus.harvest import load
from hotset.eval.tasks import freeze, generate
from hotset.eval.tasks import load as load_tasks

BATCHES, PER_BATCH, TURNS = 5, 4, 5


def main() -> None:
    catalog = load()
    sessions = load_tasks()
    seen = {s.scenario for s in sessions}
    for i in range(BATCHES):
        try:
            fresh = [
                s for s in generate(catalog, n=PER_BATCH, turns=TURNS) if s.scenario not in seen
            ]
        except Exception as exc:  # noqa: BLE001 - one bad batch must not lose the rest
            print(f"batch {i}: failed {type(exc).__name__}")
            continue
        seen.update(s.scenario for s in fresh)
        sessions.extend(fresh)
        print(f"batch {i}: +{len(fresh)} sessions")
    freeze(sessions)
    turns = [t for s in sessions for t in s.turns]
    print(
        f"total {len(sessions)} sessions / {len(turns)} turns / "
        f"{len({t.tool for t in turns})} distinct tools"
    )


if __name__ == "__main__":
    main()
