"""
Mock ticket system for the e-commerce support agent.

Tickets are the unit of work for the support agent. Each ticket has a
customer, a subject, a description, a status, and a history of comments
(including the agent's responses and any handoffs to specialist agents).
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


_VALID_STATUSES = {"open", "in_progress", "awaiting_customer", "escalated", "resolved", "closed"}
_VALID_PRIORITIES = {"low", "normal", "high", "urgent"}


class TicketSystemError(Exception):
    """Raised when the ticket system simulates a failure or rejects input."""


@dataclass
class TicketSystem:
    failure_rate: float = 0.0
    rotation_strength: float = 0.0
    latency_ms_range: tuple[int, int] = (0, 0)
    seed: int = 42
    time_anchor: datetime = field(
        default_factory=lambda: datetime(2026, 5, 19, 0, 0, 0, tzinfo=timezone.utc)
    )

    _rng: random.Random = field(init=False, repr=False)
    _tickets: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # --- Public surface --------------------------------------------------

    def create_ticket(self,
                      customer_id: str,
                      subject: str,
                      description: str,
                      priority: str = "normal") -> dict[str, Any]:
        self._maybe_fail("create_ticket")
        if priority not in _VALID_PRIORITIES:
            raise TicketSystemError(f"invalid_priority: {priority}")

        ticket_id = f"tkt_{uuid.UUID(int=self._rng.getrandbits(128)).hex[:10]}"
        now = self._now_iso()
        ticket = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "subject": subject,
            "description": description,
            "priority": priority,
            "status": "open",
            "created_at": now,
            "updated_at": now,
            "comments": [],
        }
        self._tickets[ticket_id] = ticket
        return self._serialize(ticket)

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        self._maybe_fail("get_ticket")
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise TicketSystemError(f"ticket_not_found: {ticket_id}")
        return self._serialize(ticket)

    def list_tickets(self,
                     customer_id: str | None = None,
                     status: str | None = None,
                     limit: int = 25) -> list[dict[str, Any]]:
        self._maybe_fail("list_tickets")
        results = list(self._tickets.values())
        if customer_id is not None:
            results = [t for t in results if t["customer_id"] == customer_id]
        if status is not None:
            if status not in _VALID_STATUSES:
                raise TicketSystemError(f"invalid_status_filter: {status}")
            results = [t for t in results if t["status"] == status]
        results.sort(key=lambda t: t["created_at"], reverse=True)
        return [self._serialize(t) for t in results[:limit]]

    def update_status(self, ticket_id: str, status: str) -> dict[str, Any]:
        self._maybe_fail("update_status")
        if status not in _VALID_STATUSES:
            raise TicketSystemError(f"invalid_status: {status}")
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise TicketSystemError(f"ticket_not_found: {ticket_id}")
        ticket["status"] = status
        ticket["updated_at"] = self._now_iso()
        return self._serialize(ticket)

    def add_comment(self,
                    ticket_id: str,
                    author_id: str,
                    author_role: str,
                    content: str) -> dict[str, Any]:
        """
        Add a comment from a known author. `author_role` is one of
        `customer`, `agent`, `specialist`, `system` so transcripts can
        be replayed cleanly.
        """
        self._maybe_fail("add_comment")
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise TicketSystemError(f"ticket_not_found: {ticket_id}")
        comment = {
            "comment_id": f"cmt_{uuid.UUID(int=self._rng.getrandbits(128)).hex[:10]}",
            "author_id": author_id,
            "author_role": author_role,
            "content": content,
            "created_at": self._now_iso(),
        }
        ticket["comments"].append(comment)
        ticket["updated_at"] = comment["created_at"]
        return comment

    def close_ticket(self,
                     ticket_id: str,
                     resolution: str) -> dict[str, Any]:
        self._maybe_fail("close_ticket")
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise TicketSystemError(f"ticket_not_found: {ticket_id}")
        ticket["status"] = "closed"
        ticket["resolution"] = resolution
        ticket["closed_at"] = self._now_iso()
        ticket["updated_at"] = ticket["closed_at"]
        return self._serialize(ticket)

    # --- Internal --------------------------------------------------------

    def _serialize(self, ticket: dict[str, Any]) -> dict[str, Any]:
        out = dict(ticket)
        out["comments"] = [dict(c) for c in ticket.get("comments", [])]
        if self.rotation_strength > 0 and self._rng.random() < self.rotation_strength:
            out = self._apply_rotation(out)
        return out

    def _apply_rotation(self, ticket: dict[str, Any]) -> dict[str, Any]:
        rotated = dict(ticket)
        kind = self._rng.choice(["drop_empty_comments", "echo_updated_at"])
        if kind == "drop_empty_comments" and not rotated["comments"]:
            rotated.pop("comments", None)
        elif kind == "echo_updated_at":
            # Touch the updated_at to be the same as created_at; agents
            # should not rely on micro-differences.
            rotated["updated_at"] = rotated["created_at"]
        return rotated

    def _maybe_fail(self, op: str) -> None:
        if self.failure_rate <= 0:
            return
        if self._rng.random() < self.failure_rate:
            kind = self._rng.choice(
                ["timeout", "internal_error", "service_unavailable"]
            )
            raise TicketSystemError(f"{op}_failed: {kind}")

    def _now_iso(self) -> str:
        """A deterministic timestamp anchored to time_anchor + small offsets."""
        offset = self._rng.uniform(0, 3600)
        return self.time_anchor.replace(microsecond=0).isoformat()
