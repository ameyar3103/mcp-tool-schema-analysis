"""Deterministic tool serialization — the byte-stability foundation for every cached layer."""

from __future__ import annotations

import json

from hotset.corpus.models import Tool

# Pinned separators: incidental whitespace drift shifts every downstream cache block.
_JSON_ARGS = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": False}

# JSON Schema keywords whose list order carries no meaning.
_UNORDERED_KEYS = frozenset({"required", "enum"})


def _normalize(obj, key: str | None = None):
    """Recursively stabilize a schema fragment so equal schemas serialize to equal bytes."""
    if isinstance(obj, dict):
        return {k: _normalize(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [_normalize(v) for v in obj]
        return sorted(items, key=repr) if key in _UNORDERED_KEYS else items
    return obj


def canonical_json(obj) -> str:
    """Byte-stable JSON: sorted keys, stabilized unordered lists, no whitespace variance."""
    return json.dumps(_normalize(obj), **_JSON_ARGS)


def canonical_tool(tool: Tool) -> str:
    """Full schema block for Layer B (hot set)."""
    return canonical_json(
        {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
    )


def index_line(tool: Tool) -> str:
    """One-line Layer A entry: cheap across the whole catalog, informative enough to stop hallucination."""
    required = set(tool.required_args)
    # Trailing "?" marks optional, so required args are not listed twice.
    args = ",".join(a if a in required else f"{a}?" for a in tool.arg_names)
    # Real MCP descriptions are multi-line; take the first non-blank line only.
    summary = next((ln.strip() for ln in tool.description.splitlines() if ln.strip()), "")
    return f"{tool.name}({args}) - {summary}"


def layer_a_index(tools: list[Tool]) -> str:
    """Layer A: every catalog tool, one line each, name-sorted so the block is order-stable."""
    return "\n".join(index_line(t) for t in sorted(tools, key=lambda t: t.name))
