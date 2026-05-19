"""
Mock customer database for the e-commerce support agent experiments.

This is a deliberately simple in-process API that the corresponding MCP server
wraps. It supports:

- Deterministic lookup of canonical customer records.
- Response rotation: same lookup can return slightly varied responses
  across calls (e.g., differing last_seen timestamps, occasionally missing
  optional fields). Variation rate is configurable.
- Configurable failure injection: a tunable fraction of calls return errors
  to let experiments probe agent recovery behaviour.
- Deterministic seeding so experiments are reproducible.

The "rotation" and "failure" behaviour matters for the drift experiments
because real production APIs are noisy. Holding the API perfectly stable
would understate the variance an agent has to handle.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


# --- Data model ---------------------------------------------------------

@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    tier: str  # "free" | "pro" | "enterprise"
    signup_date: str  # ISO-8601 date
    open_tickets: int = 0
    lifetime_value_usd: float = 0.0


# --- Canonical seed data -------------------------------------------------

_SEED_CUSTOMERS: list[Customer] = [
    Customer("cust_001", "Alex Rivera", "alex.rivera@example.com",
             "pro", "2024-03-14", open_tickets=1, lifetime_value_usd=482.50),
    Customer("cust_002", "Priya Anand", "priya.anand@example.com",
             "enterprise", "2023-08-02", open_tickets=0, lifetime_value_usd=15240.00),
    Customer("cust_003", "Theo Lin", "theo.lin@example.com",
             "free", "2025-11-19", open_tickets=2, lifetime_value_usd=0.0),
    Customer("cust_004", "Maya Okonkwo", "maya.o@example.com",
             "pro", "2024-12-30", open_tickets=0, lifetime_value_usd=311.20),
    Customer("cust_005", "Daichi Sato", "daichi.sato@example.com",
             "enterprise", "2022-05-11", open_tickets=3, lifetime_value_usd=48910.75),
]


# --- Error model ---------------------------------------------------------

class CustomerDBError(Exception):
    """Raised when the API simulates a failure."""


# --- Mock API class ------------------------------------------------------

@dataclass
class CustomerDB:
    """
    In-process mock of a customer database service.

    Parameters
    ----------
    failure_rate
        Probability that any given call simulates a failure (raises
        CustomerDBError). 0.0 disables failure injection.
    rotation_strength
        Probability that an otherwise successful response includes some
        rotated variation (e.g., different last_seen, missing optional
        field). 0.0 disables rotation.
    latency_ms_range
        Tuple (min, max) of milliseconds to sleep before responding,
        simulating network latency. (0, 0) disables.
    seed
        Random seed for reproducibility. Same seed + same call sequence
        produces identical results.
    """

    failure_rate: float = 0.0
    rotation_strength: float = 0.0
    latency_ms_range: tuple[int, int] = (0, 0)
    seed: int = 42
    # Anchor wall-clock used when generating mock timestamps. Holding this
    # constant makes the API fully deterministic given seed + call sequence.
    time_anchor: datetime = field(
        default_factory=lambda: datetime(2026, 5, 19, 0, 0, 0, tzinfo=timezone.utc)
    )

    _rng: random.Random = field(init=False, repr=False)
    _customers_by_id: dict[str, Customer] = field(init=False, repr=False)
    _call_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._customers_by_id = {c.customer_id: c for c in _SEED_CUSTOMERS}

    # --- Public API surface ---------------------------------------------

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """
        Look up a customer by id. Returns a serializable dict.

        Raises CustomerDBError if failure injection triggers.
        """
        self._maybe_simulate_latency()
        self._maybe_fail("get_customer")
        self._call_count += 1

        customer = self._customers_by_id.get(customer_id)
        if customer is None:
            raise CustomerDBError(f"customer_not_found: {customer_id}")

        return self._serialize(customer)

    def search_customers_by_email(self, email_fragment: str) -> list[dict[str, Any]]:
        """
        Search customers whose email contains the given fragment (case-insensitive).
        """
        self._maybe_simulate_latency()
        self._maybe_fail("search_customers_by_email")
        self._call_count += 1

        fragment = email_fragment.lower()
        matches = [c for c in self._customers_by_id.values()
                   if fragment in c.email.lower()]
        return [self._serialize(c) for c in matches]

    def list_customer_ids(self) -> list[str]:
        """Return all known customer ids. Does not fail or rotate."""
        return list(self._customers_by_id.keys())

    # --- Internal helpers -----------------------------------------------

    def _serialize(self, customer: Customer) -> dict[str, Any]:
        """Convert to dict, applying response rotation if enabled."""
        record = {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "email": customer.email,
            "tier": customer.tier,
            "signup_date": customer.signup_date,
            "open_tickets": customer.open_tickets,
            "lifetime_value_usd": customer.lifetime_value_usd,
            "last_seen": self._random_recent_timestamp(),
        }

        if self.rotation_strength > 0 and self._rng.random() < self.rotation_strength:
            record = self._apply_rotation(record)

        return record

    def _random_recent_timestamp(self) -> str:
        """A 'last_seen' timestamp somewhere in the past 30 days.

        Anchored to ``time_anchor`` so the value is fully reproducible
        given the seed and call sequence.
        """
        offset_seconds = self._rng.randint(0, 30 * 24 * 3600)
        ts = self.time_anchor - timedelta(seconds=offset_seconds)
        return ts.isoformat()

    def _apply_rotation(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Apply small benign variations: drop one optional field, jitter a
        numeric value, or rewrite the last_seen timestamp.

        Rotations never change identifying information (customer_id,
        email, name, tier). They only touch fields a reasonable agent
        should be robust to.
        """
        rotated = dict(record)
        rotation_type = self._rng.choice(["drop_optional", "jitter_ltv", "fresh_timestamp"])

        if rotation_type == "drop_optional":
            # Drop a non-essential field to test agent's robustness to absence.
            droppable = ["last_seen", "open_tickets"]
            field_to_drop = self._rng.choice(droppable)
            rotated.pop(field_to_drop, None)
        elif rotation_type == "jitter_ltv":
            # Small floating-point jitter on lifetime value (simulates
            # the kind of small inconsistencies real systems exhibit).
            rotated["lifetime_value_usd"] = round(
                rotated["lifetime_value_usd"] + self._rng.uniform(-0.5, 0.5), 2
            )
        elif rotation_type == "fresh_timestamp":
            rotated["last_seen"] = self.time_anchor.isoformat()

        return rotated

    def _maybe_simulate_latency(self) -> None:
        low, high = self.latency_ms_range
        if high <= 0:
            return
        ms = self._rng.randint(low, high)
        time.sleep(ms / 1000.0)

    def _maybe_fail(self, op: str) -> None:
        if self.failure_rate <= 0:
            return
        if self._rng.random() < self.failure_rate:
            failure_kinds = [
                "timeout",
                "internal_error",
                "rate_limit_exceeded",
                "transient_unavailable",
            ]
            kind = self._rng.choice(failure_kinds)
            raise CustomerDBError(f"{op}_failed: {kind}")
