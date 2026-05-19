"""
Tests for the 429 rate-limit retry+backoff in BaseAgent.

Uses a fake messages client that fails with RateLimitError a configurable
number of times before succeeding. The test verifies:

- The agent retries on 429 instead of bubbling the error.
- It eventually succeeds when the fake client stops failing.
- It gives up after MAX_RETRIES if the failures never stop.
- It honours a Retry-After header if present.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import Mock

import anthropic
import pytest

from src.agents.base_agent import (
    BaseAgent,
    RATE_LIMIT_MAX_RETRIES,
)


class _StubResponse:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


def _make_rate_limit_error(retry_after: str | None = None) -> anthropic.RateLimitError:
    """Construct a RateLimitError with controllable Retry-After header."""
    headers = {"retry-after": retry_after} if retry_after else {}
    response = _StubResponse(headers=headers)
    exc = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
    exc.response = response  # type: ignore[attr-defined]
    exc.message = "rate limited"
    return exc


def _patch_agent(agent: BaseAgent,
                 fail_first_n: int,
                 raise_on_call_n: int | None = None,
                 retry_after: str | None = None) -> Mock:
    """Replace the agent's anthropic client with a stub that fails N times."""
    state = {"calls": 0}

    def stub_create(**kwargs: Any) -> Any:
        state["calls"] += 1
        if state["calls"] <= fail_first_n:
            raise _make_rate_limit_error(retry_after=retry_after)
        return Mock(content=[], stop_reason="end_turn", usage=None)

    client_mock = Mock()
    client_mock.messages = Mock()
    client_mock.messages.create = stub_create
    agent._client = client_mock
    return client_mock


def _agent_for_tests() -> BaseAgent:
    return BaseAgent(
        agent_id="test_agent",
        system_prompt="test",
        toolbox=Mock(anthropic_tool_specs=Mock(return_value=[])),
    )


@pytest.mark.asyncio
async def test_retries_then_succeeds(monkeypatch):
    agent = _agent_for_tests()
    _patch_agent(agent, fail_first_n=2)

    # Skip the actual sleep to keep tests fast.
    async def _no_sleep(_s: float) -> None:
        return None
    monkeypatch.setattr("src.agents.base_agent.asyncio.sleep", _no_sleep)

    result = await agent._create_with_rate_limit_handling(model="m", max_tokens=10,
                                                          system="s", tools=[], messages=[])
    assert result is not None
    assert agent.rate_limit_retries_attempted == 2


@pytest.mark.asyncio
async def test_gives_up_after_max_retries(monkeypatch):
    agent = _agent_for_tests()
    _patch_agent(agent, fail_first_n=RATE_LIMIT_MAX_RETRIES + 5)

    async def _no_sleep(_s: float) -> None:
        return None
    monkeypatch.setattr("src.agents.base_agent.asyncio.sleep", _no_sleep)

    with pytest.raises(anthropic.RateLimitError):
        await agent._create_with_rate_limit_handling(model="m", max_tokens=10,
                                                     system="s", tools=[], messages=[])

    assert agent.rate_limit_retries_attempted == RATE_LIMIT_MAX_RETRIES


@pytest.mark.asyncio
async def test_honours_retry_after_header(monkeypatch):
    agent = _agent_for_tests()
    _patch_agent(agent, fail_first_n=1, retry_after="3")

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("src.agents.base_agent.asyncio.sleep", fake_sleep)

    await agent._create_with_rate_limit_handling(model="m", max_tokens=10,
                                                 system="s", tools=[], messages=[])

    assert sleeps == [3.0]


@pytest.mark.asyncio
async def test_zero_failures_means_zero_retries():
    agent = _agent_for_tests()
    _patch_agent(agent, fail_first_n=0)

    result = await agent._create_with_rate_limit_handling(model="m", max_tokens=10,
                                                          system="s", tools=[], messages=[])
    assert result is not None
    assert agent.rate_limit_retries_attempted == 0
