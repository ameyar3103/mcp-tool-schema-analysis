"""Launch specs for public MCP servers. Chosen to run without credentials so harvest is reproducible."""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

_SANDBOX = tempfile.gettempdir()
_REPO = Path(__file__).resolve().parents[2]
# Project-local npm cache: the global one has root-owned entries that break npx.
_NPM = {"npm_config_cache": str(_REPO / ".npm-cache")}


class ServerSpec(BaseModel):
    """How to start one MCP server over stdio."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str] = Field(default_factory=dict)


SERVERS: list[ServerSpec] = [
    ServerSpec(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", _SANDBOX],
        env=_NPM,
    ),
    ServerSpec(
        name="memory", command="npx", args=["-y", "@modelcontextprotocol/server-memory"], env=_NPM
    ),
    ServerSpec(
        name="everything",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
        env=_NPM,
    ),
    ServerSpec(
        name="sequential-thinking",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
        env=_NPM,
    ),
    ServerSpec(name="playwright", command="npx", args=["-y", "@playwright/mcp@latest"], env=_NPM),
    ServerSpec(name="git", command="uvx", args=["mcp-server-git", "--repository", str(_REPO)]),
    ServerSpec(name="fetch", command="uvx", args=["mcp-server-fetch"]),
    ServerSpec(name="time", command="uvx", args=["mcp-server-time"]),
]
