"""Three-layer prompt assembly: catalog index, hot-set schemas, ephemeral tail."""

from __future__ import annotations

import json

from hotset.corpus.models import Tool
from hotset.layout.serialize import canonical_tool, layer_a_index
from hotset.policy.base import Plan

_INTRO = "You are a tool-using assistant."

# Layers A and B render as system text, never the native tools field: only text
# blocks honour a cache breakpoint on OpenRouter (see docs/week1-findings.md).
_GUIDE_INDEX = """CATALOG lists every tool available to you, one per line, as
`name(arg, optional?) - summary`. SCHEMAS gives full parameter detail for the
tools you are most likely to need. If a CATALOG tool you need is absent from
SCHEMAS, call it anyway using the argument names shown on its catalog line.

Never invent a tool that does not appear in CATALOG."""

_GUIDE_SCHEMAS = """SCHEMAS gives full parameter detail for every tool you may call.
Never invent a tool that does not appear in SCHEMAS."""

_CALL = "To call a tool, emit a normal tool call whose name is the exact tool name."


def layer_a(catalog: list[Tool]) -> str:
    """Whole-catalog index: cheap per tool, enough to stop name hallucination."""
    return "## CATALOG\n" + layer_a_index(catalog)


def layer_b(hot: list[Tool]) -> str:
    """Full schemas for admitted tools, name-sorted so admission order never shifts bytes."""
    return "## SCHEMAS\n" + "\n".join(canonical_tool(t) for t in sorted(hot, key=lambda t: t.name))


def layer_c(tail: list[Tool]) -> str:
    """Schemas for this turn only, appended after history and dropped afterwards."""
    body = "\n".join(canonical_tool(t) for t in sorted(tail, key=lambda t: t.name))
    return "## ADDITIONAL SCHEMAS\n" + body


def _block(text: str, cached: bool) -> dict:
    """A system text block, optionally closing a cache segment."""
    block = {"type": "text", "text": text}
    return {**block, "cache_control": {"type": "ephemeral"}} if cached else block


def preamble(plan: Plan) -> str:
    """Guidance matched to the layers this plan actually renders."""
    guide = _GUIDE_INDEX if plan.index else (_GUIDE_SCHEMAS if plan.hot else "")
    # With no names in the prompt at all, telling the model to name a tool invites guessing.
    # Salt leads, so a new run invalidates the whole prefix. Without it an arm
    # re-run inside the cache TTL inherits a warm prefix it never paid for.
    head = f"[run {plan.salt}]" if plan.salt else ""
    parts = [head, _INTRO, guide, plan.instructions, _CALL if guide else ""]
    return "\n\n".join(p for p in parts if p)


def cached_prefix(plan: Plan, split: bool = True) -> list[dict]:
    """System blocks for layers A and B.

    Split gives A its own breakpoint. A is frozen and B changes on every admission,
    so where the provider honours it, admission re-writes B alone (see Q7). Pointless
    without an index, since the head would then be a few hundred bytes.
    """
    head = "\n\n".join(p for p in (preamble(plan), layer_a(plan.index) if plan.index else "") if p)
    body = layer_b(plan.hot) if plan.hot else ""
    if not split or not plan.index or not body:
        return [_block("\n\n".join(p for p in (head, body) if p), True)]
    return [_block(head, True), _block(body, True)]


# The tools field renders upstream of system, so its bytes are part of our cached
# prefix. A module constant guarantees they are identical in every arm and turn.
_DISPATCHER = {
    "type": "function",
    "function": {
        "name": "call_tool",
        "description": "Invoke one tool from the catalog. This is the only way to call a tool.",
        "parameters": {
            "type": "object",
            "properties": {
                # Free-form, not an enum: constrained decoding would make hallucinated
                # names impossible and zero out the metric we exist to measure.
                "tool": {"type": "string", "description": "Exact tool name from the catalog."},
                # Permissive: the catalog's schemas share no single argument shape.
                "arguments": {"type": "object", "description": "Arguments for that tool."},
            },
            "required": ["tool", "arguments"],
        },
    },
}


def dispatcher_tool() -> dict:
    """The one native-tools entry, kept solely to preserve structured tool_call output."""
    return _DISPATCHER


def parse_call(message: dict) -> tuple[str, dict] | None:
    """Inverse of assembly: pull (tool_name, arguments) out of a response message.

    Models usually name the catalog tool directly, having read it from cached text,
    and only sometimes wrap it in the dispatcher. Both shapes are accepted.
    """
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    fn = calls[0].get("function", {})
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    if fn.get("name") == _DISPATCHER["function"]["name"]:
        return str(args.get("tool", "")), args.get("arguments") or {}
    return fn.get("name", ""), args


def assemble(plan: Plan, messages: list[dict], split: bool = True) -> dict:
    """Request fragment: cached system prefix, then history, then the ephemeral tail."""
    msgs: list[dict] = [{"role": "system", "content": cached_prefix(plan, split)}, *messages]
    if plan.tail:
        # Suffix placement is load-bearing: dropping it next turn leaves the prefix intact.
        msgs.append({"role": "user", "content": layer_c(plan.tail)})
    tools = ([dispatcher_tool()] if plan.use_dispatcher else []) + plan.extra_tools
    return {"messages": msgs, "tools": tools}
