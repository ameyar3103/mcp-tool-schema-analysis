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
from hotset.eval.spans import Recorder, usage_attributes
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
    twin: bool = False  # named a synthetic near-duplicate of the labeled tool
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
    policy,
    spec: ModelSpec,
    catalog: list[Tool],
    session: Session,
    idx: int,
    salt: str = "",
    recorder: Recorder | None = None,
):
    """Replay one scenario in order, keeping history so the cache can actually warm."""
    names = {t.name for t in catalog}
    # A distractor derived from the labeled tool is a near-identical description, so
    # only the label distinguishes them. Scored separately, not silently as wrong.
    twins = {t.name: t.twin_of for t in catalog if t.twin_of}
    sid, history, out = uuid.uuid4().hex, [], []
    getattr(policy, "reset", lambda: None)()  # scenario boundary, not a state wipe

    rec = recorder or Recorder(policy.name, spec.slug)
    for turn_no, turn in enumerate(session.turns):
        history.append({"role": "user", "content": turn.user})
        result = TurnResult(
            arm=policy.name, model=spec.slug, session=idx, turn=turn_no, expected=turn.tool
        )
        turn_span = rec.span(
            "turn", sid, attributes={}, **{"hotset.session": idx, "hotset.turn": turn_no}
        )
        span = turn_span.__enter__()
        plan = policy.plan(catalog, history, turn.user)
        plan.salt = salt
        span.attributes["hotset.hot_size"] = len(getattr(policy, "hot", []))

        for hop in range(_MAX_HOPS):
            try:
                with rec.span("chat", sid, span.span_id, **{"hotset.hop": hop}) as call_span:
                    message, usage, elapsed = _call(spec, assemble(plan, history), sid)
                    call_span.attributes.update(usage_attributes(usage))
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
                        # Not every provider returns an id; a missing one must not lose the run.
                        "tool_call_id": message["tool_calls"][0].get("id") or f"call_{hop}",
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
            result.twin = twins.get(name, "") == turn.tool
            history.append(
                {"role": "assistant", "content": "", "tool_calls": message["tool_calls"]}
            )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_calls"][0].get("id") or f"call_{hop}",
                    "content": "OK",
                }
            )
            break

        span.attributes.update(
            {
                "hotset.expected": turn.tool,
                "hotset.predicted": result.predicted,
                "hotset.correct": result.correct,
                "hotset.twin": result.twin,
                "hotset.hallucinated": result.hallucinated,
                "hotset.hops": result.hops,
                "hotset.cost_usd": result.cost_usd,
            }
        )
        turn_span.__exit__(None, None, None)
        out.append(result)
    return out


def run_arm(
    policy,
    spec: ModelSpec,
    catalog: list[Tool],
    sessions: list[Session],
    workers: int = 4,
    salt: str = "",
    recorder: Recorder | None = None,
):
    """Sessions run in parallel; turns inside one session stay strictly ordered.

    All arms in a comparison must share one salt, so none starts warmer than another.
    """
    if getattr(policy, "stateful", False):
        workers = 1
    # One recorder per session rather than one shared across threads: a session is
    # single-threaded, so no span list is ever touched by two workers at once.
    books = [Recorder(policy.name, spec.slug) for _ in sessions]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(run_session, policy, spec, catalog, s, i, salt, books[i])
            for i, s in enumerate(sessions)
        ]
        results = [r for job in jobs for r in job.result()]
    if recorder is not None:
        for book in books:
            recorder.spans.extend(book.spans)  # merged on the main thread, after the join
    return results


def save(results: list[TurnResult], tag: str) -> Path:
    """One JSONL per arm-model pair, so partial runs are never lost."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{tag}.jsonl"
    path.write_text("".join(json.dumps(r.model_dump()) + "\n" for r in results))
    return path


def load(path: Path) -> list[TurnResult]:
    """Read an arm back for offline analysis, so tests never re-spend on the API."""
    return [TurnResult(**json.loads(line)) for line in path.read_text().splitlines() if line]


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
        "twin": sum(r.twin for r in scored) / n,
        # Credits a synthetic near-duplicate of the labeled tool. Strict accuracy asks
        # the model to guess which of two near-identical entries the label picked;
        # lenient accuracy measures whether it found the right capability.
        "lenient_accuracy": sum(r.correct or r.twin for r in scored) / n,
        "hit_rate": sum(r.hit_rate() for r in turns_with_prompt) / (len(turns_with_prompt) or 1),
        "prompt_tokens": sum(r.prompt_tokens for r in results) / n,
        "hops": sum(r.hops for r in scored) / n,
        "latency_s": sum(r.latency_s for r in scored) / n,
        "cost_per_turn": sum(r.cost_usd for r in scored) / n,
        # The metric that matters: a cheap turn that picks the wrong tool is waste.
        "cost_per_correct": (sum(r.cost_usd for r in scored) / correct if correct else math.inf),
    }
