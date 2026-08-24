"""Labeled task suite: multi-turn scenarios with a ground-truth tool per turn.

Sessions rather than isolated queries, for two reasons: a single turn cannot exhibit
cache reuse, and the week-4 Markov predictor needs tool sequences to mine. Generated
once by a strong model, then frozen to JSON so every arm sees identical inputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

from hotset.config import MODELS
from hotset.corpus.models import Tool
from hotset.layout.serialize import layer_a_index
from hotset.runtime.openrouter import pinned_body, post

SUITE = Path(__file__).resolve().parents[2] / "data" / "tasks.json"

# Queries paraphrased from tool descriptions would hand the retrieval baselines a
# free win, so the vocabulary constraint is the load-bearing part of this prompt.
_PROMPT = """Below is a catalog of tools available to an AI agent.

{index}

Invent {n} realistic multi-turn work sessions a user might have with this agent.

Rules:
- Each session is one coherent scenario with {turns} turns in a sensible order.
- Each turn is a natural request in the user's own words, stating a GOAL.
- Never reuse distinctive wording from the tool's description or its name. A user
  who does not know the tool exists must plausibly have phrased it this way.
- Exactly one catalog tool must be the correct answer for each turn.
- Vary the servers used across sessions.

Return only JSON: a list of objects
{{"scenario": str, "turns": [{{"user": str, "tool": str}}]}}"""


class Turn(BaseModel):
    """One labeled request."""

    user: str
    tool: str


class Session(BaseModel):
    """A coherent scenario, ordered."""

    scenario: str
    turns: list[Turn]


def _extract(text: str) -> list[dict]:
    """Models fence their JSON as often as not."""
    match = re.search(r"\[.*]", text, re.DOTALL)
    return json.loads(match.group(0)) if match else []


def generate(
    catalog: list[Tool], n: int = 4, turns: int = 5, model: str = "sonnet"
) -> list[Session]:
    """One batch of sessions. Invalid tool names are dropped, not repaired."""
    prompt = _PROMPT.format(index=layer_a_index(catalog), n=n, turns=turns)
    body = pinned_body(
        MODELS[model], max_tokens=4000, messages=[{"role": "user", "content": prompt}]
    )
    text = post(body)["choices"][0]["message"]["content"] or ""
    names = {t.name for t in catalog}
    out = []
    for raw in _extract(text):
        session = Session.model_validate(raw)
        session.turns = [t for t in session.turns if t.tool in names]
        if session.turns:
            out.append(session)
    return out


def freeze(sessions: list[Session], path: Path = SUITE) -> Path:
    """Write the suite sorted by scenario, so regeneration produces a readable diff."""
    ordered = sorted(sessions, key=lambda s: s.scenario)
    payload = [s.model_dump() for s in ordered]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load(path: Path = SUITE) -> list[Session]:
    """Read the frozen suite."""
    return [Session.model_validate(s) for s in json.loads(path.read_text())]


def split(sessions: list[Session], ratio: float = 0.5, seed: int = 0) -> tuple[list, list]:
    """Deterministic train/test split. Frequency priors must never see the eval set."""
    order = sorted(range(len(sessions)), key=lambda i: hash((seed, sessions[i].scenario)))
    cut = round(len(sessions) * ratio)
    return [sessions[i] for i in order[:cut]], [sessions[i] for i in order[cut:]]
