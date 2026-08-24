"""Three-layer prompt assembly: catalog index, hot-set schemas, ephemeral tail."""

from __future__ import annotations

import json

from hotset.corpus.models import Tool
from hotset.layout.serialize import canonical_tool, layer_a_index

# Layers A and B render as system text, never the native tools field: only text
# blocks honour a cache breakpoint on OpenRouter (see docs/week1-findings.md).
_PREAMBLE = """You are a tool-using assistant.

CATALOG lists every tool available to you, one per line, as
`name(arg, optional?) - summary`. SCHEMAS gives full parameter detail for the
tools you are most likely to need. If a CATALOG tool you need is absent from
SCHEMAS, call it anyway using the argument names shown on its catalog line.

Never invent a tool that does not appear in CATALOG. To call a tool, emit a
normal tool call whose name is the exact CATALOG name."""


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


def cached_prefix(
    catalog: list[Tool], hot: list[Tool], preamble: str = _PREAMBLE, split: bool = True
) -> list[dict]:
    """System blocks for layers A and B.

    Split puts a breakpoint after A as well as B. A is frozen and B changes on every
    admission, so splitting means admission re-writes only B, not the whole prefix.
    """
    head, body = "\n\n".join([preamble, layer_a(catalog)]), layer_b(hot)
    if not split:
        return [_block(head + "\n\n" + body, True)]
    return [_block(head, True), _block(body, True)]


# The tools field renders upstream of system, so its bytes are part of our cached
# prefix. A module constant guarantees they are identical in every arm and turn.
_DISPATCHER = {
    "type": "function",
    "function": {
        "name": "call_tool",
        "description": "Invoke one tool from CATALOG. This is the only way to call a tool.",
        "parameters": {
            "type": "object",
            "properties": {
                # Free-form, not an enum: constrained decoding would make hallucinated
                # names impossible and zero out the metric we exist to measure.
                "tool": {"type": "string", "description": "Exact tool name from CATALOG."},
                # Permissive: 76 heterogeneous schemas share no single argument shape.
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
    


def assemble(
    catalog: list[Tool],
    hot: list[Tool],
    messages: list[dict],
    tail: list[Tool] | None = None,
    split: bool = True,
) -> dict:
    """Request fragment: cached system prefix, then history, then the ephemeral tail."""
    prefix = cached_prefix(catalog, hot, split=split)
    msgs: list[dict] = [{"role": "system", "content": prefix}, *messages]
    if tail:
        # Suffix placement is load-bearing: dropping it next turn leaves the prefix intact.
        msgs.append({"role": "user", "content": layer_c(tail)})
    return {"messages": msgs, "tools": [dispatcher_tool()]}
