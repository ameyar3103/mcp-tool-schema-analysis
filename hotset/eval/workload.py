"""Compose an evaluation trace from the frozen suite under a controllable skew.

The suite is deliberately diverse: its generation prompt asks for varied servers, so
every tool appears a handful of times and no tool is ever hot. That is a legitimate
condition, but it is the pathological one, not the representative one — real MCP
deployments concentrate on a couple of servers and a handful of tools.

Skew is therefore an independent variable rather than a property of the corpus. The
same sessions are replayed under different arrival distributions, which isolates
workload concentration from task difficulty: the questions are identical, only how
often each kind arrives changes.
"""

from __future__ import annotations

import random
from collections import Counter

from hotset.eval.tasks import Session


def session_server(session: Session, servers: dict[str, str]) -> str:
    """A scenario's server is the modal server of its tools; ties break on name."""
    counts = Counter(servers.get(t.tool, "") for t in session.turns)
    return max(sorted(counts), key=lambda s: (counts[s], s))


def trace(
    sessions: list[Session],
    servers: dict[str, str],
    skew: float = 0.0,
    length: int = 20,
    seed: int = 0,
) -> list[Session]:
    """Sessions drawn with replacement, weighted Zipf over servers.

    skew=0 is uniform and reproduces the frozen suite's own distribution. Weight is
    per *server*, not per tool: a team adopts a server and then uses all of it, which
    is what makes a hot set exist at all.
    """
    if not sessions:
        return []
    rng = random.Random(seed)
    names = sorted({session_server(s, servers) for s in sessions})
    rng.shuffle(names)  # rank order is arbitrary, so fix it by seed rather than by name
    weight = {name: 1.0 / (rank + 1) ** skew for rank, name in enumerate(names)}
    pool = sorted(sessions, key=lambda s: s.scenario)
    return rng.choices(pool, weights=[weight[session_server(s, servers)] for s in pool], k=length)


def concentration(sessions: list[Session]) -> tuple[float, int]:
    """Top-5 share and the largest use count in any 50-turn window.

    The second number is the one that matters: admission is priced against uses per
    horizon, so a trace whose peak sits below n* cannot admit no matter the predictor.
    """
    calls = [t.tool for s in sessions for t in s.turns]
    if not calls:
        return (0.0, 0)
    top5 = sum(v for _, v in Counter(calls).most_common(5)) / len(calls)
    peak = max(
        (max(Counter(calls[i : i + 50]).values()) for i in range(max(1, len(calls) - 49))),
        default=0,
    )
    return (top5, peak)
