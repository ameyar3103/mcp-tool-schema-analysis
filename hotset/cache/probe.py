"""Week-1 gate: prove cache behaviour is observable before building any policy on it.

Every number downstream is unfalsifiable until these pass, so they run first.
"""

from __future__ import annotations

import uuid

from hotset.config import MODELS, ModelSpec
from hotset.corpus.harvest import load
from hotset.layout.serialize import as_openai_tool
from hotset.runtime.openrouter import parse_usage, pinned_body, post

_QUESTION = "Name one tool from the catalog that reads a file. Answer with the name only."


def _run(body: dict):
    """Send a request and return its cache accounting."""
    return parse_usage(post(body))


def _fresh(text: str) -> str:
    """Unique suffix so a probe never reads a cache entry an earlier probe wrote."""
    return f"{text} [{uuid.uuid4().hex[:8]}]"


def probe_tools_cacheable(spec: ModelSpec, tools: list[dict]) -> dict:
    """Q1: does cache_control on a tool definition actually cache? Layer B depends on it."""
    marked = [dict(t) for t in tools]
    marked[-1] = {**marked[-1], "cache_control": {"type": "ephemeral"}}
    system = _fresh("You are a tool-selection assistant.")

    def call():
        return _run(
            pinned_body(
                spec,
                tools=marked,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": _QUESTION},
                ],
            )
        )

    first, second = call(), call()
    return {"first": first, "second": second, "cached": second.cached_tokens > 0}


def probe_system_cacheable(spec: ModelSpec, tools: list[dict]) -> dict:
    """Control for Q1: same schemas carried as system text instead of the tools field."""
    blob = "\n".join(str(t["function"]) for t in tools)
    system = [
        {
            "type": "text",
            "text": _fresh("You are a tool-selection assistant.") + "\n" + blob,
            "cache_control": {"type": "ephemeral"},
        },
    ]

    def call():
        return _run(
            pinned_body(
                spec,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": _QUESTION},
                ],
            )
        )

    first, second = call(), call()
    return {"first": first, "second": second, "cached": second.cached_tokens > 0}


def probe_stickiness(spec: ModelSpec, tools: list[dict], turns: int = 12) -> dict:
    """Q2: does the pin hold one cache across a replay? Uses the Q1-winning layout."""
    session = uuid.uuid4().hex
    blob = "\n".join(str(t["function"]) for t in tools)
    system = [
        {
            "type": "text",
            "text": _fresh("You are a tool-selection assistant.") + "\n" + blob,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    history = [{"role": "system", "content": system}]
    rates = []
    for turn in range(turns):
        history.append({"role": "user", "content": f"{_QUESTION} (turn {turn})"})
        usage = _run(pinned_body(spec, messages=history, session_id=session))
        rates.append(usage.hit_rate())
        history.append({"role": "assistant", "content": "read_text_file"})
    return {"rates": rates, "held": all(r > 0 for r in rates[1:])}


def probe_min_cacheable(spec: ModelSpec, sizes=(256, 512, 1024, 2048, 4096, 8192)) -> dict:
    """Q3: the real floor below which caching silently no-ops."""
    found = {}
    for size in sizes:
        # "word " tokenizes near 1 token per repetition, so size approximates token count.
        system = [
            {
                "type": "text",
                "text": _fresh("ctx") + " " + ("word " * size),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": "Say ok."}]
        _run(pinned_body(spec, messages=msgs))
        found[size] = _run(pinned_body(spec, messages=msgs)).cached_tokens > 0
    return {"by_size": found, "floor": next((s for s, hit in found.items() if hit), None)}


def probe_serialization_drift(spec: ModelSpec, tools: list[dict]) -> dict:
    """Q4: reproduce the silent cache killer - logically identical, byte-different."""
    system = _fresh("You are a tool-selection assistant.")
    marked = [dict(t) for t in tools]
    marked[-1] = {**marked[-1], "cache_control": {"type": "ephemeral"}}
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": _QUESTION}]

    _run(pinned_body(spec, tools=marked, messages=msgs))
    same = _run(pinned_body(spec, tools=marked, messages=msgs))

    # Same tools, same semantics, reversed order: a pure byte-level change.
    shuffled = list(reversed(marked[:-1])) + [marked[-1]]
    drifted = _run(pinned_body(spec, tools=shuffled, messages=msgs))
    return {
        "stable": same,
        "drifted": drifted,
        "reproduced": same.cached_tokens > 0 and drifted.cached_tokens < same.cached_tokens,
    }


def main(model: str = "qwen-flash") -> None:
    spec = MODELS[model]
    tools = [as_openai_tool(t) for t in load()]
    print(f"model {spec.slug} pinned to {spec.provider}, {len(tools)} tools\n")
    spend = 0.0

    q1 = probe_tools_cacheable(spec, tools)
    print(
        f"Q1 cache_control on tools     : {'HIT' if q1['cached'] else 'MISS':4}  "
        f"write={q1['first'].write_tokens:>6,} read={q1['second'].cached_tokens:>6,} "
        f"prompt={q1['second'].prompt_tokens:,}"
    )

    q1b = probe_system_cacheable(spec, tools)
    print(
        f"Q1b same schemas as system    : {'HIT' if q1b['cached'] else 'MISS':4}  "
        f"write={q1b['first'].write_tokens:>6,} read={q1b['second'].cached_tokens:>6,} "
        f"prompt={q1b['second'].prompt_tokens:,}"
    )

    q2 = probe_stickiness(spec, tools)
    print(
        f"Q2 pin holds over {len(q2['rates'])} turns     : {'HELD' if q2['held'] else 'BROKE'}  "
        f"rates={' '.join(f'{r:.0%}' for r in q2['rates'])}"
    )

    q3 = probe_min_cacheable(spec)
    print(
        f"Q3 min cacheable prefix       : {q3['floor']} tok  "
        f"({', '.join(f'{s}:{"Y" if h else "n"}' for s, h in q3['by_size'].items())})"
    )

    q4 = probe_serialization_drift(spec, tools)
    print(
        f"Q4 reorder kills the cache    : {'REPRODUCED' if q4['reproduced'] else 'no effect':10}  "
        f"stable_read={q4['stable'].cached_tokens:>6,} after_reorder={q4['drifted'].cached_tokens:>6,}"
    )

    for r in (q1, q1b, q4):
        spend += sum(u.reported_cost_usd for u in r.values() if hasattr(u, "reported_cost_usd"))
    print(f"\nprobe spend (partial tally): ${spend:.5f}")


if __name__ == "__main__":
    main()
