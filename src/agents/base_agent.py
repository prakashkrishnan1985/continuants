"""
Base agent class.

Holds the LLM client, system prompt, MCP toolbox, and conversation state.
Provides a `run_turn` coroutine that takes a user message, lets the agent
loop through tool calls, and returns the agent's final response.

Designed to be reused by every role (primary support, technical
specialist, billing specialist). Roles differ only in their system prompt
and which MCP servers they connect to; the loop is shared.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import anthropic

from src.common.cost import CostTracker
from src.common.mcp_client import Toolbox


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_TURNS = 12  # cap on tool-call iterations per `run_turn`

# Rate-limit handling: when Anthropic returns a 429, we explicitly back off
# beyond the SDK's default retry policy. The per-minute input-token limit
# requires sustained waits (the rolling window only refreshes over a minute).
RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_BASE_BACKOFF_SECONDS = 30.0
RATE_LIMIT_MAX_BACKOFF_SECONDS = 120.0


# An extra tool is a spec dict (in Anthropic Messages API format) paired
# with an async handler that takes the LLM-supplied arguments and returns
# the tool result. Used for non-MCP tools such as A2A escalation.
ExtraToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass
class ExtraTool:
    spec: dict[str, Any]
    handler: ExtraToolHandler


@dataclass
class BaseAgent:
    agent_id: str
    system_prompt: str
    toolbox: Toolbox
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_turns: int = DEFAULT_MAX_TURNS
    extra_tools: list[ExtraTool] = field(default_factory=list)

    # Lift the SDK's built-in retries; we layer our own explicit handler on
    # top for the per-minute-token rate limit case.
    _client: anthropic.Anthropic = field(
        default_factory=lambda: anthropic.Anthropic(max_retries=4),
        init=False,
        repr=False,
    )
    _conversation: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    cost_tracker: CostTracker = field(init=False)
    rate_limit_retries_attempted: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cost_tracker = CostTracker(model=self.model)

    async def _create_with_rate_limit_handling(self, **kwargs: Any) -> Any:
        """
        Call messages.create with explicit 429 retry+backoff.

        Anthropic returns 429 when the org-level rate limit (e.g.,
        30k input tokens per minute) is hit. The SDK's built-in retries
        do not always wait long enough for the rolling window to drain.
        We add an outer loop that backs off for a full minute or more
        between attempts.
        """
        last_exc: Exception | None = None
        for attempt in range(RATE_LIMIT_MAX_RETRIES):
            try:
                return self._client.messages.create(**kwargs)
            except anthropic.RateLimitError as exc:
                last_exc = exc
                self.rate_limit_retries_attempted += 1
                retry_after = _retry_after_seconds(exc)
                if retry_after is None:
                    backoff = min(
                        RATE_LIMIT_BASE_BACKOFF_SECONDS * (2 ** attempt),
                        RATE_LIMIT_MAX_BACKOFF_SECONDS,
                    )
                    backoff += random.uniform(0, backoff * 0.1)  # 10% jitter
                else:
                    backoff = retry_after
                await asyncio.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    async def run_turn(self, user_message: str) -> str:
        """
        Send a user message, let the agent reason and call tools until
        it produces a final text response, return that response.

        The conversation persists across `run_turn` calls so that within
        a session the agent has continuity. Calling `reset()` clears it.
        """
        self._conversation.append({"role": "user", "content": user_message})
        tools = self.toolbox.anthropic_tool_specs() + [t.spec for t in self.extra_tools]
        extra_by_name = {t.spec["name"]: t.handler for t in self.extra_tools}

        for _ in range(self.max_turns):
            response = await self._create_with_rate_limit_handling(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=tools,
                messages=self._conversation,
            )

            self.cost_tracker.record_usage(getattr(response, "usage", None))
            self._conversation.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                # Final answer reached. Extract the visible text and return.
                text_parts = [
                    block.text for block in response.content
                    if getattr(block, "type", None) == "text"
                ]
                return "\n".join(text_parts).strip()

            # The agent wants to use one or more tools. Execute them
            # and append results back to the conversation.
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                qualified_name = block.name
                args = block.input or {}
                try:
                    if qualified_name in extra_by_name:
                        result = await extra_by_name[qualified_name](args)
                    else:
                        result = await self.toolbox.call(qualified_name, args)
                    content = json.dumps(result, default=str)
                    is_error = False
                except Exception as exc:  # noqa: BLE001 - tool errors must reach the agent
                    content = json.dumps({"error": f"tool_invocation_failed: {exc}"})
                    is_error = True

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                })

            self._conversation.append({"role": "user", "content": tool_results})

        # Hit the iteration cap without a final answer.
        return "[agent hit max_turns without producing a final response]"

    def reset(self) -> None:
        """Wipe conversation state for a new ticket."""
        self._conversation = []

    def conversation_snapshot(self) -> list[dict[str, Any]]:
        """Copy of the current conversation. Useful for inspectability logs."""
        return list(self._conversation)


def _retry_after_seconds(exc: anthropic.RateLimitError) -> float | None:
    """Extract `Retry-After` from a rate-limit response, if present."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
