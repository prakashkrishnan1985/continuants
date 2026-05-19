"""MCP server exposing the order_system API."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.apis.order_system import OrderSystem, OrderSystemError


_FAILURE_RATE = float(os.environ.get("ORDER_SYSTEM_FAILURE_RATE", "0.0"))
_ROTATION_STRENGTH = float(os.environ.get("ORDER_SYSTEM_ROTATION_STRENGTH", "0.0"))
_SEED = int(os.environ.get("ORDER_SYSTEM_SEED", "42"))

_svc = OrderSystem(
    failure_rate=_FAILURE_RATE,
    rotation_strength=_ROTATION_STRENGTH,
    seed=_SEED,
)

mcp = FastMCP("order_system")


@mcp.tool()
def get_order(order_id: str) -> dict[str, Any]:
    """Look up an order by id, including items and shipping info."""
    try:
        return _svc.get_order(order_id)
    except OrderSystemError as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_orders_for_customer(customer_id: str, limit: int = 25) -> list[dict[str, Any]]:
    """List recent orders placed by a customer."""
    try:
        return _svc.list_orders_for_customer(customer_id, limit=limit)
    except OrderSystemError as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def track_shipment(order_id: str) -> dict[str, Any]:
    """Return tracking status for an order."""
    try:
        return _svc.track_shipment(order_id)
    except OrderSystemError as exc:
        return {"error": str(exc)}


@mcp.tool()
def request_return(order_id: str,
                   reason: str,
                   items_to_return: list[str] | None = None) -> dict[str, Any]:
    """
    Initiate a return for a delivered order. `items_to_return` is a list
    of SKUs; pass an empty list or omit for a full-order return.
    """
    try:
        return _svc.request_return(order_id, reason, items_to_return)
    except OrderSystemError as exc:
        return {"error": str(exc)}


@mcp.tool()
def request_refund(order_id: str,
                   amount_usd: float | None = None,
                   reason: str = "") -> dict[str, Any]:
    """Request a refund on an order. Omit amount_usd for full refund."""
    try:
        return _svc.request_refund(order_id, amount_usd, reason)
    except OrderSystemError as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_refund_status(refund_id: str) -> dict[str, Any]:
    """Check the status of a previously requested refund."""
    try:
        return _svc.get_refund_status(refund_id)
    except OrderSystemError as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    mcp.run()
