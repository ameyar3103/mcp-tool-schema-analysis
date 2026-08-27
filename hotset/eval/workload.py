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

from hotset.eval.tasks import Session, Turn


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


def repeat(
    sessions: list[Session], rate: float = 0.5, seed: int = 0, cap: int = 4
) -> list[Session]:
    """Interleave extra turns that reuse a tool already called in the same session.

    Server-level skew cannot make a tool hot: concentrating every request on one server
    still spreads it over that server's dozen-odd tools, so the per-tool rate saturates
    around 12% no matter how hard the knob is turned. Admission is priced per *tool*, so
    testing it needs reuse at the tool level, which the frozen suite has almost none of.

    No new labels are invented. A repeat turn is an existing labeled turn from elsewhere
    in the suite that calls the same tool, so phrasing varies while the ground truth
    stays exactly as trustworthy as the suite it came from. Reusing the *same* turn text
    would make the repeat trivially easy and inflate every arm equally.
    """
    if not sessions or rate <= 0:
        return sessions
    rng = random.Random(seed)
    # Phrasings pool across the whole split: a tool used once here can still be repeated
    # using another session's wording for it.
    phrasings: dict[str, list[Turn]] = {}
    for s in sessions:
        for t in s.turns:
            phrasings.setdefault(t.tool, []).append(t)

    out = []
    for s in sessions:
        turns = list(s.turns)
        for _ in range(round(len(s.turns) * rate)):
            # Drawn from tools already used here, so a repeat deepens the session's own
            # working set rather than widening it.
            used = Counter(t.tool for t in turns)
            eligible = [n for n, c in used.items() if c < cap and len(phrasings[n]) > 1]
            if not eligible:
                break
            tool = rng.choice(sorted(eligible))
            options = [t for t in phrasings[tool] if t not in turns] or phrasings[tool]
            # Inserted, not appended: real reuse is interleaved, and appending would put
            # every repeat in the late turns the drift analysis reads.
            turns.insert(rng.randrange(len(turns) + 1), rng.choice(options))
        out.append(Session(scenario=s.scenario, turns=turns))
    return out
