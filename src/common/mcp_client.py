"""
Helper for connecting an agent to multiple MCP servers.

Encapsulates the stdio-transport boilerplate so agent code can focus on
the LLM loop. Each MCP server is launched as a subprocess, its tools are
discovered, and a single `Toolbox` object exposes them for an agent to
call.

This is the integration point that satisfies point 7 (inspectability):
every tool call goes through this layer and can be logged centrally.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# A toolcall logger is a callable accepting (server_name, tool_name, args, result).
# Agents can install one to record every tool invocation for inspectability.
ToolCallLogger = Callable[[str, str, dict[str, Any], Any], None]


@dataclass
class MCPServerSpec:
    """Configuration for one MCP server the agent will connect to."""

    name: str                              # logical name used in logs
    command: list[str]                     # e.g., ["python", "-m", "src.mcp_servers.customer_db_server"]
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Toolbox:
    """
    Live set of MCP-exposed tools an agent can call.

    Used as an async context manager; servers are spawned on entry and
    cleaned up on exit. Internally maintains a mapping from
    "<server>.<tool>" -> (session, tool_name).
    """

    server_specs: list[MCPServerSpec]
    logger: ToolCallLogger | None = None

    _sessions: dict[str, ClientSession] = field(default_factory=dict, init=False, repr=False)
    _tool_index: dict[str, tuple[str, str]] = field(default_factory=dict, init=False, repr=False)
    _tool_specs: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> "Toolbox":
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for spec in self.server_specs:
            params = StdioServerParameters(
                command=spec.command[0],
                args=spec.command[1:],
                env={**spec.env} if spec.env else None,
            )
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
            session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            self._sessions[spec.name] = session
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                qualified = f"{spec.name}.{tool.name}"
                self._tool_index[qualified] = (spec.name, tool.name)
                self._tool_specs.append({
                    "name": qualified,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
                })

        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._exit_stack is not None:
            await self._exit_stack.__aexit__(exc_type, exc, tb)
            self._exit_stack = None

    def anthropic_tool_specs(self) -> list[dict[str, Any]]:
        """
        Tool specs in the format Anthropic Messages API expects.

        Names use a "server.tool" convention so the agent's tool calls
        can be routed back to the right MCP server.
        """
        return [
            {
                "name": spec["name"].replace(".", "__"),
                "description": spec["description"],
                "input_schema": spec["input_schema"],
            }
            for spec in self._tool_specs
        ]

    async def call(self, qualified_name: str, arguments: dict[str, Any]) -> Any:
        """Invoke an MCP tool by its qualified `server.tool` name."""
        if "__" in qualified_name and "." not in qualified_name:
            qualified_name = qualified_name.replace("__", ".", 1)
        if qualified_name not in self._tool_index:
            raise KeyError(f"Unknown MCP tool: {qualified_name}")
        server_name, tool_name = self._tool_index[qualified_name]
        session = self._sessions[server_name]
        result = await session.call_tool(tool_name, arguments=arguments)

        payload = _unwrap_tool_result(result)

        if self.logger is not None:
            try:
                self.logger(server_name, tool_name, arguments, payload)
            except Exception:  # noqa: BLE001 - logging must never break the run
                pass
        return payload


def _unwrap_tool_result(result: Any) -> Any:
    """
    Convert an MCP CallToolResult into a JSON-friendly Python value.

    Prefers `structuredContent`, which FastMCP populates with the
    tool's original return value preserved in shape. Falls back to
    decoding text content blocks when structured content is absent.

    Important: relying on `structuredContent` avoids a subtle FastMCP
    quirk where a tool returning a single-element list is serialized
    in the text channel as just that element (losing the list shape).
    """
    if not hasattr(result, "content"):
        return result

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        # FastMCP wraps the original return value under a "result" key
        # when the tool's return type is not already a dict. If present,
        # unwrap it; otherwise return the dict as-is.
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured

    pieces: list[Any] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is None:
            pieces.append(block)
            continue
        try:
            pieces.append(json.loads(text))
        except json.JSONDecodeError:
            pieces.append(text)
    if len(pieces) == 1:
        return pieces[0]
    return pieces
