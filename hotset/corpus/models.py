"""Typed representation of an MCP tool, normalized across servers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Tool(BaseModel):
    """One catalog entry, as returned by an MCP server's tools/list."""

    name: str
    description: str = ""
    input_schema: dict = Field(default_factory=dict)
    server: str = ""  # provenance: which MCP server exposed it
    synthetic: bool = False  # generated near-duplicate distractor
    twin_of: str = ""  # for a distractor, the real tool it was derived from

    @property
    def arg_names(self) -> list[str]:
        """Declared argument names, schema order preserved."""
        return list(self.input_schema.get("properties", {}))

    @property
    def required_args(self) -> list[str]:
        """Args the schema marks required."""
        return list(self.input_schema.get("required", []))
