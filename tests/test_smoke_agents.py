"""
Smoke test: agents wire up to MCP servers without crashing.

This test does not hit the Anthropic API. It just verifies that an
agent's toolbox can launch its MCP servers, discover tools, and route a
direct tool call through the MCP plumbing.

Skipped if the mcp package is unavailable.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from src.common.mcp_client import MCPServerSpec, Toolbox


@pytest.mark.asyncio
async def test_customer_db_toolbox_can_call_through_mcp(tmp_path):
    python = sys.executable
    specs = [
        MCPServerSpec(
            name="customer_db",
            command=[python, "-m", "src.mcp_servers.customer_db_server"],
        ),
    ]
    async with Toolbox(server_specs=specs) as toolbox:
        tools = toolbox.anthropic_tool_specs()
        names = {t["name"] for t in tools}
        assert "customer_db__get_customer" in names
        assert "customer_db__list_customer_ids" in names

        ids = await toolbox.call("customer_db.list_customer_ids", {})
        assert "cust_001" in ids

        record = await toolbox.call("customer_db.get_customer", {"customer_id": "cust_001"})
        assert record["customer_id"] == "cust_001"
        assert record["name"] == "Alex Rivera"


@pytest.mark.asyncio
async def test_memory_toolbox_persists_writes(tmp_path):
    python = sys.executable
    mem_file = tmp_path / "agent_memory.json"
    specs = [
        MCPServerSpec(
            name="memory",
            command=[python, "-m", "src.mcp_servers.memory_server"],
            env={"MEMORY_STORE_PATH": str(mem_file)},
        ),
    ]
    async with Toolbox(server_specs=specs) as toolbox:
        entry_id = await toolbox.call("memory.write_memory", {
            "body": "Customer cust_001 reported a damaged shipment; resolved with replacement.",
            "tags": ["damaged", "replacement"],
            "context": {"customer_id": "cust_001"},
        })
        assert isinstance(entry_id, str)
        assert entry_id.startswith("mem_")

        recent = await toolbox.call("memory.list_recent_memories", {"limit": 5})
        assert len(recent) == 1
        assert recent[0]["id"] == entry_id

    assert mem_file.exists()
