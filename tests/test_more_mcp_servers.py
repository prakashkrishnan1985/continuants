"""
Smoke tests verifying ticket_system and order_system MCP servers expose
their tools correctly and round-trip data.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("mcp")

from src.common.mcp_client import MCPServerSpec, Toolbox


@pytest.mark.asyncio
async def test_ticket_system_via_mcp_full_lifecycle(tmp_path):
    specs = [
        MCPServerSpec(
            name="ticket_system",
            command=[sys.executable, "-m", "src.mcp_servers.ticket_system_server"],
        ),
    ]
    async with Toolbox(server_specs=specs) as toolbox:
        names = {t["name"] for t in toolbox.anthropic_tool_specs()}
        assert {
            "ticket_system__create_ticket",
            "ticket_system__get_ticket",
            "ticket_system__list_tickets",
            "ticket_system__update_ticket_status",
            "ticket_system__add_ticket_comment",
            "ticket_system__close_ticket",
        }.issubset(names)

        ticket = await toolbox.call(
            "ticket_system.create_ticket",
            {
                "customer_id": "cust_001",
                "subject": "Order has not arrived",
                "description": "It has been 10 days since I placed order ord_a1b2.",
                "priority": "high",
            },
        )
        assert ticket["status"] == "open"
        tid = ticket["ticket_id"]

        comment = await toolbox.call(
            "ticket_system.add_ticket_comment",
            {
                "ticket_id": tid,
                "author_id": "primary_support_01",
                "author_role": "agent",
                "content": "Looking into this now.",
            },
        )
        assert comment["author_role"] == "agent"

        updated = await toolbox.call(
            "ticket_system.update_ticket_status",
            {"ticket_id": tid, "status": "in_progress"},
        )
        assert updated["status"] == "in_progress"

        closed = await toolbox.call(
            "ticket_system.close_ticket",
            {"ticket_id": tid, "resolution": "Provided shipment update and ETA."},
        )
        assert closed["status"] == "closed"

        fetched = await toolbox.call("ticket_system.get_ticket", {"ticket_id": tid})
        assert fetched["status"] == "closed"
        assert fetched["resolution"] == "Provided shipment update and ETA."


@pytest.mark.asyncio
async def test_order_system_via_mcp(tmp_path):
    specs = [
        MCPServerSpec(
            name="order_system",
            command=[sys.executable, "-m", "src.mcp_servers.order_system_server"],
        ),
    ]
    async with Toolbox(server_specs=specs) as toolbox:
        names = {t["name"] for t in toolbox.anthropic_tool_specs()}
        for expected in (
            "order_system__get_order",
            "order_system__list_orders_for_customer",
            "order_system__track_shipment",
            "order_system__request_return",
            "order_system__request_refund",
            "order_system__get_refund_status",
        ):
            assert expected in names

        order = await toolbox.call("order_system.get_order", {"order_id": "ord_a1b2"})
        assert order["customer_id"] == "cust_001"
        assert order["status"] == "delivered"

        orders = await toolbox.call(
            "order_system.list_orders_for_customer",
            {"customer_id": "cust_001"},
        )
        assert len(orders) >= 1
        assert all(o["customer_id"] == "cust_001" for o in orders)

        tracking = await toolbox.call(
            "order_system.track_shipment",
            {"order_id": "ord_c3d4"},
        )
        assert tracking["status"] in {"in_transit", "out_for_delivery"}

        refund = await toolbox.call(
            "order_system.request_refund",
            {"order_id": "ord_a1b2", "reason": "wrong color"},
        )
        assert refund["status"] == "approved"
        assert refund["amount_usd"] > 0

        refund_status = await toolbox.call(
            "order_system.get_refund_status",
            {"refund_id": refund["refund_id"]},
        )
        assert refund_status["refund_id"] == refund["refund_id"]
