"""Harvest tools/list from real MCP servers and freeze the catalog to versioned JSON.

Freezing matters: index lines emit args in schema order, so an upstream server
reordering its JSON would shift Layer A bytes and miss the whole prefix.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from hotset.corpus.models import Tool
from hotset.corpus.servers import SERVERS, ServerSpec

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "catalog.json"
_TIMEOUT_S = 120


async def harvest_server(spec: ServerSpec) -> list[Tool]:
    """Start one server, list its tools, shut it down."""
    params = StdioServerParameters(
        command=spec.command, args=spec.args, env={**os.environ, **spec.env}
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = await session.list_tools()
    return [
        Tool(
            name=t.name,
            description=t.description or "",
            input_schema=t.input_schema or {},
            server=spec.name,
        )
        for t in listed.tools
    ]


async def harvest_all(specs: list[ServerSpec] | None = None) -> list[Tool]:
    """Harvest every server, tolerating individual failures so one bad server can't block the corpus."""
    tools: list[Tool] = []
    for spec in specs or SERVERS:
        try:
            found = await asyncio.wait_for(harvest_server(spec), timeout=_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {spec.name:22} FAILED  {type(exc).__name__}: {str(exc)[:70]}")
            continue
        print(f"  {spec.name:22} {len(found):>3} tools")
        tools.extend(found)
    return tools


def freeze(tools: list[Tool], path: Path = CATALOG_PATH) -> Path:
    """Write the catalog name-sorted with stable formatting so diffs are meaningful."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(tools, key=lambda t: (t.server, t.name))
    path.write_text(json.dumps([t.model_dump() for t in ordered], indent=2, sort_keys=True) + "\n")
    return path


def load(path: Path = CATALOG_PATH) -> list[Tool]:
    """Read the frozen catalog."""
    return [Tool(**d) for d in json.loads(path.read_text())]


if __name__ == "__main__":
    harvested = asyncio.run(harvest_all())
    out = freeze(harvested)
    print(f"\n{len(harvested)} tools from {len({t.server for t in harvested})} servers -> {out}")
