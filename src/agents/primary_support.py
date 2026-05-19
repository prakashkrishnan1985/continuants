"""
Primary support agent.

A BaseAgent wired to:
  - customer_db MCP server
  - knowledge_base MCP server
  - memory MCP server (per-agent isolated memory file)
  - ticket_system MCP server
  - order_system MCP server

Optionally also presented an A2A "escalate" tool so the LLM can hand
off to a specialist agent through the A2A fabric.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from src.a2a.client import A2AClient
from src.a2a.integration import make_escalate_extra_tool
from src.agents.base_agent import BaseAgent, ExtraTool
from src.agents.system_prompts import primary_support_system_prompt
from src.common.mcp_client import MCPServerSpec, Toolbox, ToolCallLogger


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def primary_server_specs(memory_path: Path) -> list[MCPServerSpec]:
    python = sys.executable
    return [
        MCPServerSpec(
            name="customer_db",
            command=[python, "-m", "src.mcp_servers.customer_db_server"],
        ),
        MCPServerSpec(
            name="knowledge_base",
            command=[python, "-m", "src.mcp_servers.knowledge_base_server"],
        ),
        MCPServerSpec(
            name="memory",
            command=[python, "-m", "src.mcp_servers.memory_server"],
            env={"MEMORY_STORE_PATH": str(memory_path.resolve())},
        ),
        MCPServerSpec(
            name="ticket_system",
            command=[python, "-m", "src.mcp_servers.ticket_system_server"],
        ),
        MCPServerSpec(
            name="order_system",
            command=[python, "-m", "src.mcp_servers.order_system_server"],
        ),
    ]


def make_primary_support_agent(agent_id: str = "primary_support_01",
                               memory_path: Path | None = None,
                               logger: ToolCallLogger | None = None,
                               model: str | None = None,
                               a2a_client: Optional[A2AClient] = None,
                               technical_specialist_id: str = "tech_specialist_01",
                               billing_specialist_id: str | None = None,
                               ) -> tuple[BaseAgent, Toolbox]:
    """
    Construct a primary support agent.

    If `a2a_client` is provided, the agent gets `escalate_to_technical_specialist`
    (and `escalate_to_billing_specialist` if billing_specialist_id is set)
    as extra tools, routed via A2A.
    """
    memory_path = memory_path or (PROJECT_ROOT / "memory" / "primary_memory.json")

    toolbox = Toolbox(
        server_specs=primary_server_specs(memory_path),
        logger=logger,
    )

    extra_tools: list[ExtraTool] = []
    if a2a_client is not None:
        extra_tools.append(make_escalate_extra_tool(
            client=a2a_client,
            tool_name="escalate_to_technical_specialist",
            recipient_id=technical_specialist_id,
            capability="diagnose_technical_issue",
            description=(
                "Escalate a customer support case to the technical specialist "
                "agent for deeper diagnosis. Use this when the issue is a "
                "product bug, integration failure, account-recovery edge case, "
                "or anything requiring technical investigation beyond your "
                "direct scope. Pass the full customer context so the "
                "specialist does not have to re-discover it."
            ),
        ))
        if billing_specialist_id is not None:
            extra_tools.append(make_escalate_extra_tool(
                client=a2a_client,
                tool_name="escalate_to_billing_specialist",
                recipient_id=billing_specialist_id,
                capability="resolve_billing_dispute",
                description=(
                    "Escalate a billing or refund dispute to the billing "
                    "specialist agent. Use this for refund authorization "
                    "decisions, billing-cycle questions, or payment-method "
                    "disputes that require specialist judgment."
                ),
            ))

    agent = BaseAgent(
        agent_id=agent_id,
        system_prompt=primary_support_system_prompt(),
        toolbox=toolbox,
        model=model or "claude-sonnet-4-6",
        extra_tools=extra_tools,
    )
    return agent, toolbox
