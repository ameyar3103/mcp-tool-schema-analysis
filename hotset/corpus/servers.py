"""Launch specs for public MCP servers. Chosen to run without credentials so harvest is reproducible."""

from __future__ import annotations

import tempfile

from pydantic import BaseModel, Field

_SANDBOX = tempfile.gettempdir()


class ServerSpec(BaseModel):
    """How to start one MCP server over stdio."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str] = Field(default_factory=dict)


SERVERS: list[ServerSpec] = [
    ServerSpec(name="filesystem", command="npx",
               args=["-y", "@modelcontextprotocol/server-filesystem", _SANDBOX]),
    ServerSpec(name="memory", command="npx",
               args=["-y", "@modelcontextprotocol/server-memory"]),
    ServerSpec(name="everything", command="npx",
               args=["-y", "@modelcontextprotocol/server-everything"]),
    ServerSpec(name="sequential-thinking", command="npx",
               args=["-y", "@modelcontextprotocol/server-sequential-thinking"]),
    ServerSpec(name="playwright", command="npx", args=["-y", "@playwright/mcp@latest"]),
    ServerSpec(name="git", command="uvx", args=["mcp-server-git", "--repository", _SANDBOX]),
    ServerSpec(name="fetch", command="uvx", args=["mcp-server-fetch"]),
    ServerSpec(name="time", command="uvx", args=["mcp-server-time"]),
]
