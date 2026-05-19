"""
MCP server exposing the agent's memory store.

Run as a standalone stdio MCP server:
  python -m src.mcp_servers.memory_server

The memory file path is read from MEMORY_STORE_PATH environment variable
so different agents (primary, technical specialist, billing specialist)
can each have their own memory file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.apis.memory_store import MemoryStore


_PATH = Path(os.environ.get("MEMORY_STORE_PATH", "./memory/agent_memory.json"))
_store = MemoryStore(path=_PATH)

mcp = FastMCP("memory")


@mcp.tool()
def write_memory(body: str,
                 tags: list[str] | None = None,
                 context: dict[str, Any] | None = None) -> str:
    """
    Write a structured note to the agent's memory.

    `body` is the lesson or observation. `tags` are short labels used for
    later retrieval. `context` is an optional dict describing the situation
    in which the note was made (used for generalization discipline: only
    re-use a note when the context matches).
    """
    return _store.write(body=body, tags=tags, context=context)


@mcp.tool()
def search_memory(tag: str | None = None,
                  text: str | None = None,
                  limit: int = 10) -> list[dict[str, Any]]:
    """
    Search memory entries. Pass `tag` for tag filter or `text` for
    substring match on the body. Results are most-recent-first.
    """
    return _store.search(tag=tag, text=text, limit=limit)


@mcp.tool()
def list_recent_memories(limit: int = 5) -> list[dict[str, Any]]:
    """Most recent memory entries, regardless of tag or content."""
    return _store.list_all()[:limit]


@mcp.tool()
def memory_count() -> int:
    """How many memory entries are currently stored."""
    return _store.count()


if __name__ == "__main__":
    mcp.run()
