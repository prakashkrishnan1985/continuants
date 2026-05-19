"""
Tests for the workflow control arm.

Verifies that the no-op memory MCP server exposes the same tool surface
as the real memory server, but performs no persistence.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("mcp")

from src.common.mcp_client import MCPServerSpec, Toolbox


@pytest.mark.asyncio
async def test_noop_memory_exposes_same_tool_surface_as_real(tmp_path):
    specs = [
        MCPServerSpec(
            name="memory",
            command=[sys.executable, "-m", "src.mcp_servers.noop_memory_server"],
        ),
    ]
    async with Toolbox(server_specs=specs) as toolbox:
        names = {t["name"] for t in toolbox.anthropic_tool_specs()}
        assert {
            "memory__write_memory",
            "memory__search_memory",
            "memory__list_recent_memories",
            "memory__memory_count",
        }.issubset(names)


@pytest.mark.asyncio
async def test_noop_memory_writes_do_not_persist(tmp_path):
    specs = [
        MCPServerSpec(
            name="memory",
            command=[sys.executable, "-m", "src.mcp_servers.noop_memory_server"],
        ),
    ]
    async with Toolbox(server_specs=specs) as toolbox:
        # Several writes succeed (return an id) but do not persist.
        eid_1 = await toolbox.call("memory.write_memory", {
            "body": "first note that should not stick",
        })
        eid_2 = await toolbox.call("memory.write_memory", {
            "body": "second note that should not stick",
            "tags": ["whatever"],
        })
        assert isinstance(eid_1, str) and eid_1.startswith("noop_mem_")
        assert isinstance(eid_2, str) and eid_2.startswith("noop_mem_")
        assert eid_1 != eid_2

        # All reads return empty.
        assert await toolbox.call("memory.search_memory", {}) == []
        assert await toolbox.call("memory.list_recent_memories", {"limit": 50}) == []
        assert await toolbox.call("memory.memory_count", {}) == 0
