"""
MCP server exposing the knowledge_base API.

Run as a standalone stdio MCP server:
  python -m src.mcp_servers.knowledge_base_server
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.apis.knowledge_base import KnowledgeBase, KnowledgeBaseError


_FAILURE_RATE = float(os.environ.get("KB_FAILURE_RATE", "0.0"))
_ROTATION_STRENGTH = float(os.environ.get("KB_ROTATION_STRENGTH", "0.0"))
_SEED = int(os.environ.get("KB_SEED", "42"))

_kb = KnowledgeBase(
    failure_rate=_FAILURE_RATE,
    rotation_strength=_ROTATION_STRENGTH,
    seed=_SEED,
)

mcp = FastMCP("knowledge_base")


@mcp.tool()
def search_kb(query: str | None = None,
              tag: str | None = None,
              limit: int = 5) -> list[dict[str, Any]]:
    """
    Search the knowledge base. Pass `query` for free-text search on
    titles and bodies, `tag` to filter by tag, or both.
    """
    try:
        return _kb.search(query=query, tag=tag, limit=limit)
    except KnowledgeBaseError as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def get_kb_entry(entry_id: str) -> dict[str, Any]:
    """Retrieve a specific KB entry by id."""
    try:
        return _kb.get(entry_id)
    except KnowledgeBaseError as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_kb_tags() -> list[str]:
    """List every tag in the knowledge base."""
    return _kb.list_tags()


if __name__ == "__main__":
    mcp.run()
