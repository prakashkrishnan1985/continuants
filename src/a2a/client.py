"""
A2A client adapter used by agents.

The router and registry are the protocol layer. An A2AClient is the thin
wrapper an agent uses to:

- Register its own AgentCard at startup.
- Register handlers for each capability it serves.
- Send requests to other agents and receive their replies.

The client also exposes an "as_anthropic_tool" helper so the primary
agent's LLM can call A2A escalation as if it were a normal tool. This
keeps the LLM-facing interface uniform between MCP tools and A2A
escalations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from src.a2a.protocol import (
    A2AHandler,
    A2AMessage,
    A2ARegistry,
    A2ARouter,
    AgentCard,
)


@dataclass
class A2AClient:
    """An agent's view onto the A2A fabric."""

    agent_id: str
    registry: A2ARegistry
    router: A2ARouter

    def register_self(self, card: AgentCard) -> None:
        """Publish this agent's card so peers can discover it."""
        if card.agent_id != self.agent_id:
            raise ValueError("AgentCard.agent_id must match A2AClient.agent_id")
        self.registry.register(card)

    def register_handler(self,
                         capability: str,
                         handler: A2AHandler) -> None:
        """Handle inbound requests on the named capability."""
        self.router.register_handler(self.agent_id, capability, handler)

    def find_capable_peers(self, capability: str) -> list[AgentCard]:
        """Discover other agents that advertise this capability."""
        return [
            card for card in self.registry.find_by_capability(capability)
            if card.agent_id != self.agent_id
        ]

    async def request(self,
                      recipient_id: str,
                      capability: str,
                      payload: dict[str, Any]) -> dict[str, Any]:
        """Send a request and return the reply payload."""
        return await self.router.request(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            capability=capability,
            payload=payload,
        )
