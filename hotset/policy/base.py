"""Policy interface: each policy decides what lands in layers A, B and C per turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from hotset.corpus.models import Tool


@dataclass
class Plan:
    """One turn's layout request. Empty layers are simply omitted from the prompt."""

    index: list[Tool] = field(default_factory=list)  # layer A, cached one-liners
    hot: list[Tool] = field(default_factory=list)  # layer B, cached full schemas
    tail: list[Tool] = field(default_factory=list)  # layer C, ephemeral suffix
    extra_tools: list[dict] = field(default_factory=list)  # native tools past the dispatcher
    instructions: str = ""  # policy-specific guidance appended to the preamble
    use_dispatcher: bool = True  # drop it when an arm supplies its own native tool
    salt: str = ""  # per-run nonce; the runner sets it, policies never do


class Policy(Protocol):
    """Anything that can pick tools for a turn. Stateless policies ignore history."""

    name: str

    def plan(self, catalog: list[Tool], history: list[dict], query: str) -> Plan: ...
