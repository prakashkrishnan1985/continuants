"""MCP server exposing the ticket_system API."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.apis.ticket_system import TicketSystem, TicketSystemError


_FAILURE_RATE = float(os.environ.get("TICKET_SYSTEM_FAILURE_RATE", "0.0"))
_ROTATION_STRENGTH = float(os.environ.get("TICKET_SYSTEM_ROTATION_STRENGTH", "0.0"))
_SEED = int(os.environ.get("TICKET_SYSTEM_SEED", "42"))

_svc = TicketSystem(
    failure_rate=_FAILURE_RATE,
    rotation_strength=_ROTATION_STRENGTH,
    seed=_SEED,
)

mcp = FastMCP("ticket_system")


@mcp.tool()
def create_ticket(customer_id: str,
                  subject: str,
                  description: str,
                  priority: str = "normal") -> dict[str, Any]:
    """Create a new support ticket. `priority` is one of low|normal|high|urgent."""
    try:
        return _svc.create_ticket(customer_id, subject, description, priority)
    except TicketSystemError as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_ticket(ticket_id: str) -> dict[str, Any]:
    """Fetch a ticket by id, including its comment history."""
    try:
        return _svc.get_ticket(ticket_id)
    except TicketSystemError as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_tickets(customer_id: str | None = None,
                 status: str | None = None,
                 limit: int = 25) -> list[dict[str, Any]]:
    """List tickets, optionally filtered by customer_id and/or status."""
    try:
        return _svc.list_tickets(customer_id=customer_id, status=status, limit=limit)
    except TicketSystemError as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def update_ticket_status(ticket_id: str, status: str) -> dict[str, Any]:
    """Move a ticket to a new status (open, in_progress, awaiting_customer, escalated, resolved, closed)."""
    try:
        return _svc.update_status(ticket_id, status)
    except TicketSystemError as exc:
        return {"error": str(exc)}


@mcp.tool()
def add_ticket_comment(ticket_id: str,
                       author_id: str,
                       author_role: str,
                       content: str) -> dict[str, Any]:
    """Add a comment to a ticket. `author_role` is one of customer|agent|specialist|system."""
    try:
        return _svc.add_comment(ticket_id, author_id, author_role, content)
    except TicketSystemError as exc:
        return {"error": str(exc)}


@mcp.tool()
def close_ticket(ticket_id: str, resolution: str) -> dict[str, Any]:
    """Close a ticket with a resolution summary."""
    try:
        return _svc.close_ticket(ticket_id, resolution)
    except TicketSystemError as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    mcp.run()
