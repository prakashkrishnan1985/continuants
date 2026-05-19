"""
No-op memory MCP server.

Exposes the same tool surface as the real memory server, but every
operation is a pass-through that does not persist or retrieve anything.
Used by the control (workflow) arm of the drift experiment so the
agent's tool surface is identical to the treatment arm but no state
accumulates.

This is what makes the control arm a proper "workflow" by our 9-point
definition: it has no realization of point 2 (directed learning from
experience). The control agent's memory tool calls all succeed but
never produce any persistent effect.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("memory")


@mcp.tool()
def write_memory(body: str,
                 tags: list[str] | None = None,
                 context: dict[str, Any] | None = None) -> str:
    """
    Pretends to write a memory entry. Returns a freshly minted id so
    the calling agent does not detect failure, but no persistence
    happens. Used by the workflow control arm of the experiment.
    """
    return f"noop_mem_{uuid.uuid4().hex[:12]}"


@mcp.tool()
def search_memory(tag: str | None = None,
                  text: str | None = None,
                  limit: int = 10) -> list[dict[str, Any]]:
    """Returns an empty list. Workflow control agents have no memory to search."""
    return []


@mcp.tool()
def list_recent_memories(limit: int = 5) -> list[dict[str, Any]]:
    return []


@mcp.tool()
def memory_count() -> int:
    return 0


if __name__ == "__main__":
    mcp.run()
