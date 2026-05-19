"""
Technical specialist agent.

Receives escalations from the primary support agent via A2A. Has fewer
MCP tools than the primary because its role is narrower: it does not
need to discover arbitrary customers, only to deal with the case that
was handed off.

Tools:
  - knowledge_base (search technical KB entries)
  - memory (own memory file, isolated from the primary's)
  - ticket_system (update ticket status, add comments)
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.agents.system_prompts import technical_specialist_system_prompt
from src.common.mcp_client import MCPServerSpec, Toolbox, ToolCallLogger


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def technical_specialist_server_specs(memory_path: Path) -> list[MCPServerSpec]:
    python = sys.executable
    return [
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
    ]


def make_technical_specialist_agent(agent_id: str = "tech_specialist_01",
                                    memory_path: Path | None = None,
                                    logger: ToolCallLogger | None = None,
                                    model: str | None = None) -> tuple[BaseAgent, Toolbox]:
    memory_path = memory_path or (PROJECT_ROOT / "memory" / "technical_specialist_memory.json")
    toolbox = Toolbox(
        server_specs=technical_specialist_server_specs(memory_path),
        logger=logger,
    )
    agent = BaseAgent(
        agent_id=agent_id,
        system_prompt=technical_specialist_system_prompt(),
        toolbox=toolbox,
        model=model or "claude-sonnet-4-6",
    )
    return agent, toolbox
