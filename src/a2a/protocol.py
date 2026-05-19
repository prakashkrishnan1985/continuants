"""
Minimal Agent-to-Agent (A2A) protocol implementation.

Inspired by Google's A2A specification but kept intentionally small so
the experiments can attribute observed behaviour to the agents rather
than to protocol quirks. The pieces we need:

- AgentCard: how each agent declares its identity and capabilities.
- A2AMessage: the wire format used between agents.
- A2ARegistry: discovers agents by capability.
- A2ARouter: routes a message from a sender to one receiver, awaits the
  reply, returns it.

Production A2A includes streaming, long-running tasks, push notifications,
artefact channels, etc. We deliberately omit all of these for the pilot.
What we keep is enough for the primary support agent to escalate a ticket
to a specialist agent and receive a structured reply.

All A2A traffic flowing through the router is logged for inspectability,
matching point 7 of the agent definition.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


# --- Agent identity & capabilities --------------------------------------

@dataclass
class AgentCard:
    """
    Declares an agent's identity and what it can do.

    `capabilities` is a list of named skills the agent advertises (for
    example: "diagnose_technical_issue", "process_refund"). Other agents
    discover capable peers by querying the registry against these names.
    """

    agent_id: str
    name: str
    version: str
    capabilities: list[str]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Wire format ---------------------------------------------------------

@dataclass
class A2AMessage:
    """
    A single A2A message exchanged between agents.

    `kind` is `request` for a message that expects a reply, or `response`
    for the reply. `correlation_id` ties responses to their original
    request.
    """

    sender_id: str
    recipient_id: str
    kind: str                            # "request" | "response"
    capability: str                      # which advertised capability this targets
    payload: dict[str, Any]
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    correlation_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "kind": self.kind,
            "capability": self.capability,
            "payload": self.payload,
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }


# --- Registry ------------------------------------------------------------

class A2ARegistry:
    """In-process registry of agents and their advertised capabilities."""

    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}

    def register(self, card: AgentCard) -> None:
        self._cards[card.agent_id] = card

    def unregister(self, agent_id: str) -> None:
        self._cards.pop(agent_id, None)

    def get(self, agent_id: str) -> AgentCard | None:
        return self._cards.get(agent_id)

    def find_by_capability(self, capability: str) -> list[AgentCard]:
        """Return every registered agent that advertises the given capability."""
        return [c for c in self._cards.values() if capability in c.capabilities]

    def all_cards(self) -> list[AgentCard]:
        return list(self._cards.values())


# --- Router --------------------------------------------------------------

# A handler is what an agent registers with the router to respond to
# inbound requests on a given capability. It takes the inbound message
# and returns the reply payload (dict).
A2AHandler = Callable[[A2AMessage], Awaitable[dict[str, Any]]]

# Logger receives every message that flows through the router (in either
# direction) for inspectability.
A2AMessageLogger = Callable[[A2AMessage], None]


class A2ARouter:
    """
    Routes messages between agents in the same process.

    Each agent registers a handler per capability it serves. The router
    delivers inbound messages to the right handler, awaits the reply,
    and returns it to the caller as a response message.
    """

    def __init__(self,
                 registry: A2ARegistry,
                 logger: A2AMessageLogger | None = None) -> None:
        self.registry = registry
        self.logger = logger
        # Map (agent_id, capability) -> handler
        self._handlers: dict[tuple[str, str], A2AHandler] = {}

    def register_handler(self,
                         agent_id: str,
                         capability: str,
                         handler: A2AHandler) -> None:
        self._handlers[(agent_id, capability)] = handler

    def unregister_handler(self, agent_id: str, capability: str) -> None:
        self._handlers.pop((agent_id, capability), None)

    async def request(self,
                      sender_id: str,
                      recipient_id: str,
                      capability: str,
                      payload: dict[str, Any],
                      timeout_seconds: float = 60.0) -> dict[str, Any]:
        """
        Send a request from `sender_id` to `recipient_id` on the named
        capability, await the reply, and return the reply payload.
        """
        sender_card = self.registry.get(sender_id)
        recipient_card = self.registry.get(recipient_id)
        if recipient_card is None:
            raise ValueError(f"a2a_recipient_not_registered: {recipient_id}")
        if capability not in recipient_card.capabilities:
            raise ValueError(
                f"a2a_recipient_does_not_advertise_capability: "
                f"{recipient_id} cannot {capability}"
            )

        handler = self._handlers.get((recipient_id, capability))
        if handler is None:
            raise ValueError(
                f"a2a_no_handler_registered: agent={recipient_id} capability={capability}"
            )

        request_msg = A2AMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            kind="request",
            capability=capability,
            payload=payload,
        )
        self._log(request_msg)

        try:
            reply_payload = await asyncio.wait_for(handler(request_msg), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"a2a_handler_timeout: {recipient_id} did not reply within "
                f"{timeout_seconds}s on capability {capability}"
            )

        response_msg = A2AMessage(
            sender_id=recipient_id,
            recipient_id=sender_id,
            kind="response",
            capability=capability,
            payload=reply_payload,
            correlation_id=request_msg.message_id,
        )
        self._log(response_msg)

        # Sender_id may not actually be registered if the message is from
        # a non-agent caller (e.g., the experiment harness). That's fine,
        # we only require the recipient to be registered.
        _ = sender_card

        return reply_payload

    def _log(self, message: A2AMessage) -> None:
        if self.logger is None:
            return
        try:
            self.logger(message)
        except Exception:  # noqa: BLE001
            pass
