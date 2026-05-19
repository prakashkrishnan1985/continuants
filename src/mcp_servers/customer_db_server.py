"""
MCP server exposing the customer_db API.

Run as a standalone stdio MCP server: `python -m src.mcp_servers.customer_db_server`.
Connects to MCP clients (e.g., the support agent) over stdin/stdout.

Tool surface exposed:
  - get_customer(customer_id) -> dict
  - search_customers_by_email(email_fragment) -> list[dict]
  - list_customer_ids() -> list[str]
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.apis.customer_db import CustomerDB, CustomerDBError


# Config is read from environment so experiments can vary failure rate and
# rotation strength without code changes.
_FAILURE_RATE = float(os.environ.get("CUSTOMER_DB_FAILURE_RATE", "0.0"))
_ROTATION_STRENGTH = float(os.environ.get("CUSTOMER_DB_ROTATION_STRENGTH", "0.0"))
_SEED = int(os.environ.get("CUSTOMER_DB_SEED", "42"))

_db = CustomerDB(
    failure_rate=_FAILURE_RATE,
    rotation_strength=_ROTATION_STRENGTH,
    seed=_SEED,
)

mcp = FastMCP("customer_db")


@mcp.tool()
def get_customer(customer_id: str) -> dict[str, Any]:
    """
    Retrieve a customer record by id.

    Returns customer profile fields. Raises an error string if the
    customer is not found or if the service simulates a failure.
    """
    try:
        return _db.get_customer(customer_id)
    except CustomerDBError as exc:
        return {"error": str(exc)}


@mcp.tool()
def search_customers_by_email(email_fragment: str) -> list[dict[str, Any]]:
    """
    Find customers whose email contains the given fragment (case-insensitive).
    """
    try:
        return _db.search_customers_by_email(email_fragment)
    except CustomerDBError as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def list_customer_ids() -> list[str]:
    """List every known customer id. Deterministic, does not fail."""
    return _db.list_customer_ids()


if __name__ == "__main__":
    mcp.run()
