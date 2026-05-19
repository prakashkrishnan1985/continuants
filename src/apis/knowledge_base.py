"""
Mock knowledge base for the e-commerce support agent.

Contains synthetic FAQs and policy entries the agent can search when
responding to customers. Like the customer DB, this supports response
rotation and failure injection so experiments can probe agent robustness.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any


# --- Seed knowledge base content -----------------------------------------

_SEED_ENTRIES: list[dict[str, Any]] = [
    {
        "id": "kb_returns_policy",
        "title": "Returns and refunds policy",
        "tags": ["returns", "refunds", "policy"],
        "body": (
            "Customers may return unused items within 30 days of delivery for a "
            "full refund. Items must be in original packaging. Refunds are "
            "processed within 5-7 business days to the original payment method. "
            "Final-sale items and personalized products are non-refundable."
        ),
    },
    {
        "id": "kb_shipping_options",
        "title": "Shipping options and delivery times",
        "tags": ["shipping", "delivery", "policy"],
        "body": (
            "Standard shipping: 5-7 business days, free on orders over $50. "
            "Express shipping: 2-3 business days, $12.99. "
            "Overnight: next business day, $24.99. "
            "International shipping available to 40+ countries; delivery times vary."
        ),
    },
    {
        "id": "kb_order_tracking",
        "title": "Tracking your order",
        "tags": ["shipping", "tracking", "orders"],
        "body": (
            "Once an order ships, customers receive an email with a tracking "
            "number. Tracking can also be viewed in the 'My Orders' section "
            "after signing in. If tracking shows no movement for more than 5 "
            "business days, contact support."
        ),
    },
    {
        "id": "kb_account_billing",
        "title": "Billing and payment methods",
        "tags": ["billing", "payment", "account"],
        "body": (
            "We accept major credit cards, PayPal, Apple Pay, and Google Pay. "
            "Pro tier subscriptions are billed monthly; Enterprise is billed "
            "annually. Customers can update payment methods in account settings."
        ),
    },
    {
        "id": "kb_damaged_items",
        "title": "Damaged or defective items",
        "tags": ["returns", "damage", "policy"],
        "body": (
            "If an item arrives damaged or defective, customers should contact "
            "support within 14 days with photos. We will arrange a replacement "
            "at no cost or issue a full refund, customer's choice."
        ),
    },
    {
        "id": "kb_cancellation",
        "title": "Cancelling an order",
        "tags": ["orders", "cancellation", "policy"],
        "body": (
            "Orders can be cancelled within 1 hour of placement if they have "
            "not yet entered fulfillment. After fulfillment begins, customers "
            "must wait for delivery and then initiate a return."
        ),
    },
    {
        "id": "kb_account_recovery",
        "title": "Account recovery and password reset",
        "tags": ["account", "password", "support"],
        "body": (
            "Customers who cannot access their account can use the 'forgot "
            "password' link on the sign-in page. If the registered email is "
            "no longer accessible, support requires identity verification "
            "before transferring the account to a new email."
        ),
    },
]


class KnowledgeBaseError(Exception):
    """Raised when the KB simulates a failure."""


@dataclass
class KnowledgeBase:
    failure_rate: float = 0.0
    rotation_strength: float = 0.0
    latency_ms_range: tuple[int, int] = (0, 0)
    seed: int = 42

    _rng: random.Random = field(init=False, repr=False)
    _entries_by_id: dict[str, dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._entries_by_id = {e["id"]: e for e in _SEED_ENTRIES}

    # --- Public surface --------------------------------------------------

    def search(self,
               query: str | None = None,
               tag: str | None = None,
               limit: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve KB entries matching either a free-text query (substring
        on title or body) or a tag filter. Both may be provided.
        """
        self._maybe_simulate_latency()
        self._maybe_fail("search")

        results = list(self._entries_by_id.values())
        if tag is not None:
            results = [e for e in results if tag in e.get("tags", [])]
        if query is not None:
            q = query.lower()
            results = [
                e for e in results
                if q in e["title"].lower() or q in e["body"].lower()
            ]
        results = results[:limit]
        return [self._serialize(e) for e in results]

    def get(self, entry_id: str) -> dict[str, Any]:
        """Fetch a specific entry by id."""
        self._maybe_simulate_latency()
        self._maybe_fail("get")

        entry = self._entries_by_id.get(entry_id)
        if entry is None:
            raise KnowledgeBaseError(f"kb_entry_not_found: {entry_id}")
        return self._serialize(entry)

    def list_tags(self) -> list[str]:
        """All tags across the KB. Deterministic, does not fail."""
        seen: set[str] = set()
        for entry in self._entries_by_id.values():
            seen.update(entry.get("tags", []))
        return sorted(seen)

    # --- Internal --------------------------------------------------------

    def _serialize(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Copy and maybe rotate."""
        out = dict(entry)
        out["tags"] = list(entry.get("tags", []))
        if self.rotation_strength > 0 and self._rng.random() < self.rotation_strength:
            out = self._apply_rotation(out)
        return out

    def _apply_rotation(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Light rotation: shuffle tag order or trim body whitespace differently."""
        rotated = dict(entry)
        kind = self._rng.choice(["shuffle_tags", "trim_body_whitespace"])
        if kind == "shuffle_tags":
            tags = list(rotated["tags"])
            self._rng.shuffle(tags)
            rotated["tags"] = tags
        elif kind == "trim_body_whitespace":
            rotated["body"] = " ".join(rotated["body"].split())
        return rotated

    def _maybe_simulate_latency(self) -> None:
        low, high = self.latency_ms_range
        if high <= 0:
            return
        time.sleep(self._rng.randint(low, high) / 1000.0)

    def _maybe_fail(self, op: str) -> None:
        if self.failure_rate <= 0:
            return
        if self._rng.random() < self.failure_rate:
            kind = self._rng.choice(
                ["timeout", "internal_error", "service_unavailable"]
            )
            raise KnowledgeBaseError(f"{op}_failed: {kind}")
