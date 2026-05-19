"""
Bridges between A2A protocol and BaseAgent.

Provides:

- `make_escalate_extra_tool(client, capability, recipient_id, description)`
  builds an ExtraTool the primary agent can present to its LLM as a
  normal tool. When the LLM calls it, the tool routes the message
  through A2A to the named recipient and returns the reply.

- `register_agent_as_handler(client, agent, capability, ...)` registers
  a BaseAgent as the handler for a given A2A capability. Inbound
  requests are passed to the agent's `run_turn`, the agent's response
  is wrapped as the A2A reply payload.
"""

from __future__ import annotations

from typing import Any

from src.a2a.client import A2AClient
from src.a2a.protocol import A2AMessage
from src.agents.base_agent import BaseAgent, ExtraTool


def make_escalate_extra_tool(client: A2AClient,
                             tool_name: str,
                             recipient_id: str,
                             capability: str,
                             description: str) -> ExtraTool:
    """
    Build an ExtraTool that escalates via A2A.

    The LLM sees one parameter, `escalation_message`, and an optional
    `context` dict. The tool sends the message via A2A and returns the
    specialist's reply payload.
    """

    spec = {
        "name": tool_name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "escalation_message": {
                    "type": "string",
                    "description": (
                        "The full message to send to the specialist agent. "
                        "Include all relevant context: the customer's "
                        "request, anything you've already tried, what you "
                        "specifically want them to address."
                    ),
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Optional structured context (e.g., customer_id, "
                        "order_id, ticket_id) the specialist should have."
                    ),
                },
            },
            "required": ["escalation_message"],
        },
    }

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return await client.request(
            recipient_id=recipient_id,
            capability=capability,
            payload={
                "message": args["escalation_message"],
                "context": args.get("context") or {},
            },
        )

    return ExtraTool(spec=spec, handler=handler)


def register_agent_as_handler(client: A2AClient,
                              agent: BaseAgent,
                              capability: str,
                              reset_conversation_per_request: bool = False) -> None:
    """
    Register a BaseAgent to handle inbound A2A requests on a capability.

    When a peer sends a request, the agent receives the message and any
    context as its user prompt and produces a reply through its normal
    `run_turn` loop. The reply text and the agent's tool-call trace are
    returned as the A2A response payload.
    """

    async def handler(message: A2AMessage) -> dict[str, Any]:
        if reset_conversation_per_request:
            agent.reset()

        body = message.payload.get("message", "")
        ctx = message.payload.get("context") or {}
        formatted = body
        if ctx:
            formatted = (
                f"Incoming A2A request from {message.sender_id}.\n"
                f"Context: {ctx}\n\n"
                f"Request:\n{body}"
            )

        reply_text = await agent.run_turn(formatted)
        return {
            "responder_id": agent.agent_id,
            "reply": reply_text,
            "correlation_id": message.message_id,
        }

    client.register_handler(capability, handler)
