"""
Workflow control arm.

Constructs the *control* (workflow) counterpart to the primary support
agent. Same model, same system prompt, same MCP tool surface; the only
differences are:

- the memory MCP server is the no-op variant, so no state accumulates
  in the agent's notes;
- the conversation is reset between every ticket (handled by the
  experiment runner, not at construction time), so no context carries
  forward beyond a single ticket;
- no A2A escalation. The control arm does not escalate to specialists;
  it must handle every ticket within its own scope. This keeps the
  experimental contrast clean: treatment has learning machinery,
  control does not.

Per our agent definition, this is a workflow rather than an agent. The
experiment tests whether the agent (treatment) drifts while the
workflow (control) does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.agents.system_prompts import primary_support_system_prompt
from src.common.mcp_client import MCPServerSpec, Toolbox, ToolCallLogger


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def workflow_control_server_specs() -> list[MCPServerSpec]:
    """
    Same as primary_support_server_specs except for the memory server,
    which is replaced with the no-op variant.
    """
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
            command=[python, "-m", "src.mcp_servers.noop_memory_server"],
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


def make_workflow_control_agent(agent_id: str = "workflow_control_01",
                                logger: ToolCallLogger | None = None,
                                model: str | None = None) -> tuple[BaseAgent, Toolbox]:
    """
    Construct the workflow control agent.

    Note: this constructor does NOT install A2A escalation. The control
    arm intentionally lacks the ability to hand off to specialists; any
    escalation pathway would let the system accumulate state across
    agents and contaminate the comparison.

    The experiment runner is responsible for calling `agent.reset()`
    between every ticket so no conversation context survives the ticket
    boundary.
    """
    toolbox = Toolbox(
        server_specs=workflow_control_server_specs(),
        logger=logger,
    )
    agent = BaseAgent(
        agent_id=agent_id,
        system_prompt=primary_support_system_prompt(),
        toolbox=toolbox,
        model=model or "claude-sonnet-4-6",
    )
    return agent, toolbox
