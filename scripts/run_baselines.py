"""Run the comparison arms over the frozen task suite and print the metric vector."""

from __future__ import annotations

import sys
import uuid

from hotset.config import MODELS
from hotset.corpus.distractors import pad_catalog
from hotset.corpus.harvest import load
from hotset.eval.runner import run_arm, save, summarize
from hotset.eval.tasks import load as load_tasks
from hotset.policy.baselines import FullCatalog, LazyDiscovery, RagOverTools

HEAD = f"{'arm':16} {'acc':>6} {'halluc':>7} {'hit':>6} {'prompt':>8} {'hops':>5} {'lat':>6} {'$/turn':>9} {'$/correct':>10}"


def main(model: str = "qwen-flash", size: int = 76) -> None:
    spec = MODELS[model]
    base = load()
    catalog = base if size == len(base) else pad_catalog(base, size, seed=0)
    sessions = load_tasks()
    salt = uuid.uuid4().hex[:8]  # one per run, shared by every arm
    print(f"{model} | catalog {len(catalog)} | {len(sessions)} sessions | salt {salt}\n\n{HEAD}")

    for arm in (FullCatalog(), RagOverTools(k=8), LazyDiscovery()):
        results = run_arm(arm, spec, catalog, sessions, salt=salt)
        save(results, f"{arm.name}-{model}-{len(catalog)}-{salt}")
        m = summarize(results)
        print(
            f"{arm.name:16} {m['accuracy']:6.1%} {m['hallucinated']:7.1%} {m['hit_rate']:6.1%} "
            f"{m['prompt_tokens']:8.0f} {m['hops']:5.2f} {m['latency_s']:5.1f}s "
            f"${m['cost_per_turn']:.6f} ${m['cost_per_correct']:.6f}"
            + (f"  ({m['errors']} err)" if m["errors"] else "")
        )


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []), **({"size": int(sys.argv[2])} if len(sys.argv) > 2 else {}))
