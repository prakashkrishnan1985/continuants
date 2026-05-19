"""
Mock order system for the e-commerce support agent.

Supports orders, shipment tracking, returns, and refunds. Each operation
can be configured to inject failures and rotate responses.

This is intentionally chunky enough to give the agent something realistic
to do (lookup customer, look up their order, check shipment, initiate
return, process refund) without ballooning the codebase.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


_VALID_ORDER_STATUSES = {
    "placed", "fulfilling", "shipped", "out_for_delivery",
    "delivered", "cancelled", "returned",
}
_VALID_REFUND_STATUSES = {"requested", "approved", "processing", "completed", "denied"}


class OrderSystemError(Exception):
    """Raised when the order system fails or rejects input."""


# --- Seed data -----------------------------------------------------------

def _seed_orders() -> dict[str, dict[str, Any]]:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    orders: list[dict[str, Any]] = [
        {
            "order_id": "ord_a1b2",
            "customer_id": "cust_001",
            "items": [{"sku": "PRD-001", "name": "Wireless headphones", "qty": 1, "price": 129.00}],
            "subtotal_usd": 129.00,
            "shipping_usd": 0.00,
            "total_usd": 129.00,
            "status": "delivered",
            "placed_at": (base + timedelta(days=5)).isoformat(),
            "delivered_at": (base + timedelta(days=10)).isoformat(),
            "tracking_number": "1Z9999W66603425623",
            "shipping_address": "123 Main St, Boston, MA 02108",
        },
        {
            "order_id": "ord_c3d4",
            "customer_id": "cust_001",
            "items": [
                {"sku": "PRD-013", "name": "USB-C cable", "qty": 2, "price": 9.99},
                {"sku": "PRD-077", "name": "Phone case", "qty": 1, "price": 24.50},
            ],
            "subtotal_usd": 44.48,
            "shipping_usd": 7.99,
            "total_usd": 52.47,
            "status": "shipped",
            "placed_at": (base + timedelta(days=30)).isoformat(),
            "tracking_number": "1Z9999W66603425624",
            "shipping_address": "123 Main St, Boston, MA 02108",
        },
        {
            "order_id": "ord_e5f6",
            "customer_id": "cust_002",
            "items": [{"sku": "PRD-044", "name": "Ergonomic keyboard", "qty": 1, "price": 249.99}],
            "subtotal_usd": 249.99,
            "shipping_usd": 0.00,
            "total_usd": 249.99,
            "status": "out_for_delivery",
            "placed_at": (base + timedelta(days=33)).isoformat(),
            "tracking_number": "1Z9999W66603425625",
            "shipping_address": "788 Oak Ave, Brooklyn, NY 11215",
        },
        {
            "order_id": "ord_g7h8",
            "customer_id": "cust_003",
            "items": [{"sku": "PRD-091", "name": "Standing desk mat", "qty": 1, "price": 79.99}],
            "subtotal_usd": 79.99,
            "shipping_usd": 7.99,
            "total_usd": 87.98,
            "status": "fulfilling",
            "placed_at": (base + timedelta(days=40)).isoformat(),
            "shipping_address": "55 Cherry Ln, Austin, TX 78701",
        },
        {
            "order_id": "ord_i9j0",
            "customer_id": "cust_005",
            "items": [
                {"sku": "PRD-002", "name": "4K monitor", "qty": 1, "price": 549.00},
                {"sku": "PRD-101", "name": "HDMI cable 6ft", "qty": 1, "price": 14.99},
            ],
            "subtotal_usd": 563.99,
            "shipping_usd": 0.00,
            "total_usd": 563.99,
            "status": "delivered",
            "placed_at": (base + timedelta(days=12)).isoformat(),
            "delivered_at": (base + timedelta(days=15)).isoformat(),
            "tracking_number": "1Z9999W66603425626",
            "shipping_address": "44 Pine Rd, Seattle, WA 98109",
        },
    ]
    return {o["order_id"]: o for o in orders}


# --- Mock service --------------------------------------------------------

@dataclass
class OrderSystem:
    failure_rate: float = 0.0
    rotation_strength: float = 0.0
    seed: int = 42
    time_anchor: datetime = field(
        default_factory=lambda: datetime(2026, 5, 19, 0, 0, 0, tzinfo=timezone.utc)
    )

    _rng: random.Random = field(init=False, repr=False)
    _orders: dict[str, dict[str, Any]] = field(init=False, repr=False)
    _refunds: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _returns: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._orders = _seed_orders()

    # --- Orders ----------------------------------------------------------

    def get_order(self, order_id: str) -> dict[str, Any]:
        self._maybe_fail("get_order")
        order = self._orders.get(order_id)
        if order is None:
            raise OrderSystemError(f"order_not_found: {order_id}")
        return self._serialize(order)

    def list_orders_for_customer(self, customer_id: str, limit: int = 25) -> list[dict[str, Any]]:
        self._maybe_fail("list_orders_for_customer")
        results = [o for o in self._orders.values() if o["customer_id"] == customer_id]
        results.sort(key=lambda o: o["placed_at"], reverse=True)
        return [self._serialize(o) for o in results[:limit]]

    # --- Shipment tracking -----------------------------------------------

    def track_shipment(self, order_id: str) -> dict[str, Any]:
        self._maybe_fail("track_shipment")
        order = self._orders.get(order_id)
        if order is None:
            raise OrderSystemError(f"order_not_found: {order_id}")

        status = order["status"]
        if status in {"placed", "fulfilling"}:
            return {"order_id": order_id, "status": "not_yet_shipped"}
        if status == "shipped":
            return {
                "order_id": order_id,
                "status": "in_transit",
                "tracking_number": order.get("tracking_number"),
                "carrier": "UPS",
                "estimated_delivery": (self.time_anchor + timedelta(days=2)).date().isoformat(),
                "last_scan_location": "Memphis, TN",
            }
        if status == "out_for_delivery":
            return {
                "order_id": order_id,
                "status": "out_for_delivery",
                "tracking_number": order.get("tracking_number"),
                "carrier": "UPS",
                "estimated_delivery": self.time_anchor.date().isoformat(),
            }
        if status == "delivered":
            return {
                "order_id": order_id,
                "status": "delivered",
                "tracking_number": order.get("tracking_number"),
                "delivered_at": order.get("delivered_at"),
            }
        return {"order_id": order_id, "status": status, "note": "no tracking applicable"}

    # --- Returns and refunds --------------------------------------------

    def request_return(self,
                       order_id: str,
                       reason: str,
                       items_to_return: list[str] | None = None) -> dict[str, Any]:
        """
        Initiate a return for an order. `items_to_return` is a list of SKUs;
        empty means full order.
        """
        self._maybe_fail("request_return")
        order = self._orders.get(order_id)
        if order is None:
            raise OrderSystemError(f"order_not_found: {order_id}")
        if order["status"] != "delivered":
            raise OrderSystemError(f"cannot_return_in_status: {order['status']}")

        return_id = f"ret_{uuid.UUID(int=self._rng.getrandbits(128)).hex[:10]}"
        record = {
            "return_id": return_id,
            "order_id": order_id,
            "customer_id": order["customer_id"],
            "reason": reason,
            "items": items_to_return or [item["sku"] for item in order["items"]],
            "status": "approved",
            "created_at": self.time_anchor.isoformat(),
        }
        self._returns[return_id] = record
        return record

    def request_refund(self,
                       order_id: str,
                       amount_usd: float | None = None,
                       reason: str = "") -> dict[str, Any]:
        self._maybe_fail("request_refund")
        order = self._orders.get(order_id)
        if order is None:
            raise OrderSystemError(f"order_not_found: {order_id}")
        if amount_usd is None:
            amount_usd = order["total_usd"]
        if amount_usd <= 0 or amount_usd > order["total_usd"]:
            raise OrderSystemError(f"invalid_refund_amount: {amount_usd}")

        refund_id = f"rfn_{uuid.UUID(int=self._rng.getrandbits(128)).hex[:10]}"
        record = {
            "refund_id": refund_id,
            "order_id": order_id,
            "customer_id": order["customer_id"],
            "amount_usd": round(amount_usd, 2),
            "reason": reason,
            "status": "approved",
            "created_at": self.time_anchor.isoformat(),
        }
        self._refunds[refund_id] = record
        return record

    def get_refund_status(self, refund_id: str) -> dict[str, Any]:
        self._maybe_fail("get_refund_status")
        record = self._refunds.get(refund_id)
        if record is None:
            raise OrderSystemError(f"refund_not_found: {refund_id}")
        return dict(record)

    # --- Internal --------------------------------------------------------

    def _serialize(self, order: dict[str, Any]) -> dict[str, Any]:
        out = dict(order)
        out["items"] = [dict(i) for i in order["items"]]
        if self.rotation_strength > 0 and self._rng.random() < self.rotation_strength:
            out = self._apply_rotation(out)
        return out

    def _apply_rotation(self, order: dict[str, Any]) -> dict[str, Any]:
        rotated = dict(order)
        kind = self._rng.choice(["jitter_subtotal", "trim_shipping_address"])
        if kind == "jitter_subtotal":
            rotated["subtotal_usd"] = round(order["subtotal_usd"] + self._rng.uniform(-0.01, 0.01), 2)
        elif kind == "trim_shipping_address":
            rotated["shipping_address"] = order["shipping_address"].strip()
        return rotated

    def _maybe_fail(self, op: str) -> None:
        if self.failure_rate <= 0:
            return
        if self._rng.random() < self.failure_rate:
            kind = self._rng.choice(["timeout", "internal_error", "service_unavailable"])
            raise OrderSystemError(f"{op}_failed: {kind}")
