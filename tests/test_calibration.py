"""Calibration replay: the admitted/predicted/actual bookkeeping must be honest."""

from __future__ import annotations

import sys
from pathlib import Path

from hotset.config import MODELS
from hotset.corpus.models import Tool
from hotset.eval.tasks import Session, Turn
from hotset.policy.adaptive import HotSet
from hotset.policy.economics import asymptotic_rate
from hotset.policy.predictors import LRUK, Oracle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calibration import HORIZON, admissions

SPEC = MODELS["haiku"]
BAR = asymptotic_rate(SPEC) * HORIZON


def catalog(n: int) -> list[Tool]:
    """Schemas large enough that admission is a real cost, not a rounding error."""
    return [
        Tool(
            name=f"t{i}",
            description=f"tool number {i} for the calibration fixture",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        for i in range(n)
    ]


def sessions(names: list[str], per: int = 5) -> list[Session]:
    """One flat trace chopped into fixed-length sessions."""
    turns = [Turn(user=f"do {n}", tool=n) for n in names]
    return [Session(scenario="s", turns=turns[i : i + per]) for i in range(0, len(turns), per)]


def replay(predictor, names: list[str], size: int) -> list[tuple[float, int]]:
    return admissions(
        HotSet(SPEC, predictor, horizon=HORIZON), catalog(size), sessions(names), names
    )


def test_oracle_admits_only_tools_that_earn():
    """The oracle sees the future, so every admission clears the floor by construction."""
    names = ["t0"] * 30 + [f"t{i % 8}" for i in range(30)]
    rows = replay(Oracle(names), names, 8)
    assert rows, "a trace this concentrated must admit something"
    assert all(actual >= BAR for _, actual in rows)


def test_flat_trace_admits_nothing_under_oracle():
    """No tool repeats often enough to earn a schema, so a perfect forecast buys none."""
    names = [f"t{i}" for i in range(60)]
    assert replay(Oracle(names), names, 60) == []


def test_lruk_overforecasts_a_burst():
    """A tight burst extrapolates across the whole horizon; the trace does not deliver."""
    names = ["t0", "t0", "t0"] + [f"t{i % 20}" for i in range(1, 60)]
    rows = replay(LRUK(k=2), names, 20)
    assert rows, "the burst must trip admission at all"
    assert sum(p for p, _ in rows) > 2 * sum(a for _, a in rows)


def test_each_tool_admitted_once():
    """A tool held hot for many turns is one admission, not one per turn it survives."""
    names = ["t0"] * 40
    rows = replay(Oracle(names), names, 6)
    assert len(rows) == 1
