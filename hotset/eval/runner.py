"""Play one arm over the frozen task suite, recording the metrics vector per turn."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from hotset.config import ModelSpec
from hotset.corpus.models import Tool
from hotset.eval.tasks import Session
from hotset.layout.prompt import assemble, parse_call
from hotset.runtime.openrouter import parse_usage, pinned_body, post

RESULTS = Path(__file__).resolve().parents[2] / "results"
_MAX_HOPS = 4  # lazy discovery can search repeatedly; cap it so a loop cannot run away


class TurnResult(BaseModel):
    """One labeled turn under one arm. Everything the Pareto plot needs."""

    arm: str
    model: str
    session: int
    turn: int
    expected: str
    predicted: str = ""
    correct: bool = False
    hallucinated: bool = False  # named a tool that is not in the catalog
    hops: int = 0  # extra round trips before the real call
    cached: int = 0
    written: int = 0
    uncached: int = 0
    output: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str = ""  # transport failure, so it is not scored as a wrong answer

    @property
    def prompt_tokens(self) -> int:
        """Writes count: a cold turn genuinely did send those bytes."""
        return self.cached + self.written + self.uncached

    def hit_rate(self) -> float:
        """Share of prompt tokens served from cache."""
        return self.cached / self.prompt_tokens if self.prompt_tokens else 0.0


def _call(spec: ModelSpec, frag: dict, session_id: str) -> tuple[dict, object, float]:
    """One request, timed. Latency here is wall clock, not TTFT: streaming comes later."""
    start = time.perf_counter()
    payload = post(pinned_body(spec, max_tokens=256, session_id=session_id, **frag))
    return payload["choices"][0]["message"], parse_usage(payload), time.perf_counter() - start


def run_session(
    policy, spec: ModelSpec, catalog: list[Tool], session: Session, idx: int, salt: str = ""
):
    """Replay one scenario in order, keeping history so the cache can actually warm."""
    names = {t.name for t in catalog}
    sid, history, out = uuid.uuid4().hex, [], []

    for turn_no, turn in enumerate(session.turns):
        history.append({"role": "user", "content": turn.user})
        result = TurnResult(
            arm=policy.name, model=spec.slug, session=idx, turn=turn_no, expected=turn.tool
        )
        plan = policy.plan(catalog, history, turn.user)
        plan.salt = salt

        for hop in range(_MAX_HOPS):
            try:
                message, usage, elapsed = _call(spec, assemble(plan, history), sid)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                break
            result.cached += usage.cached_tokens
            result.written += usage.write_tokens
            result.uncached += usage.uncached_tokens
            result.output += usage.output_tokens
            result.cost_usd += usage.total_cost_usd(spec)
            result.latency_s += elapsed

            call = parse_call(message)
            if call is None:
                break
            name, args = call
            # A policy-served tool (lazy discovery's registry) is a hop, not an answer.
            if getattr(policy, "serves", lambda _: False)(name):
                history.append(
                    {"role": "assistant", "content": "", "tool_calls": message["tool_calls"]}
                )
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_calls"][0]["id"],
                        "content": policy.serve(catalog, args),
                    }
                )
                result.hops = hop + 1
                continue
            result.predicted = name
            # Deployments only see what was called, so that is what the predictor gets.
            getattr(policy, "observe", lambda _: None)(name)
            result.correct = name == turn.tool
            result.hallucinated = bool(name) and name not in names
            history.append(
                {"role": "assistant", "content": "", "tool_calls": message["tool_calls"]}
            )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_calls"][0]["id"],
                    "content": "OK",
                }
            )
            break

        out.append(result)
    return out


def run_arm(
    policy,
    spec: ModelSpec,
    catalog: list[Tool],
    sessions: list[Session],
    workers: int = 4,
    salt: str = "",
):
    """Sessions run in parallel; turns inside one session stay strictly ordered.

    All arms in a comparison must share one salt, so none starts warmer than another.
    """
    if getattr(policy, "stateful", False):
        workers = 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(run_session, policy, spec, catalog, s, i, salt)
            for i, s in enumerate(sessions)
        ]
        return [r for job in jobs for r in job.result()]


def save(results: list[TurnResult], tag: str) -> Path:
    """One JSONL per arm-model pair, so partial runs are never lost."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{tag}.jsonl"
    path.write_text("".join(json.dumps(r.model_dump()) + "\n" for r in results))
    return path


def summarize(results: list[TurnResult]) -> dict:
    """The headline metric vector for one arm."""
    scored = [r for r in results if not r.error]
    correct = sum(r.correct for r in scored)
    n = len(scored) or 1
    turns_with_prompt = [r for r in scored if r.prompt_tokens]
    return {
        "turns": len(scored),
        "errors": len(results) - len(scored),
        "accuracy": sum(r.correct for r in scored) / n,
        "hallucinated": sum(r.hallucinated for r in scored) / n,
        "hit_rate": sum(r.hit_rate() for r in turns_with_prompt) / (len(turns_with_prompt) or 1),
        "prompt_tokens": sum(r.prompt_tokens for r in results) / n,
        "hops": sum(r.hops for r in scored) / n,
        "latency_s": sum(r.latency_s for r in scored) / n,
        "cost_per_turn": sum(r.cost_usd for r in scored) / n,
        # The metric that matters: a cheap turn that picks the wrong tool is waste.
        "cost_per_correct": (sum(r.cost_usd for r in scored) / correct if correct else math.inf),
    }
