"""
Tests for the A2A protocol.

These tests do not hit the Anthropic API. They register simple async
handlers as "agents" and verify the protocol layer routes messages,
discovers capabilities, and surfaces errors correctly.
"""

from __future__ import annotations

import pytest

from src.a2a import (
    A2AClient,
    A2AMessage,
    A2ARegistry,
    A2ARouter,
    AgentCard,
)


def _build_fabric() -> tuple[A2ARegistry, A2ARouter, list[A2AMessage]]:
    log: list[A2AMessage] = []
    registry = A2ARegistry()
    router = A2ARouter(registry=registry, logger=log.append)
    return registry, router, log


def _make_client(agent_id: str,
                 registry: A2ARegistry,
                 router: A2ARouter,
                 capabilities: list[str],
                 name: str | None = None) -> A2AClient:
    client = A2AClient(agent_id=agent_id, registry=registry, router=router)
    client.register_self(AgentCard(
        agent_id=agent_id,
        name=name or agent_id,
        version="1.0.0",
        capabilities=capabilities,
    ))
    return client


@pytest.mark.asyncio
async def test_register_and_discover_by_capability():
    registry, router, _ = _build_fabric()
    _make_client("alpha", registry, router, capabilities=["foo", "bar"])
    _make_client("beta", registry, router, capabilities=["baz"])
    _make_client("gamma", registry, router, capabilities=["foo"])

    foos = registry.find_by_capability("foo")
    assert {c.agent_id for c in foos} == {"alpha", "gamma"}

    bazs = registry.find_by_capability("baz")
    assert {c.agent_id for c in bazs} == {"beta"}

    missing = registry.find_by_capability("nonexistent")
    assert missing == []


@pytest.mark.asyncio
async def test_request_routes_to_correct_handler_and_returns_reply():
    registry, router, log = _build_fabric()
    sender = _make_client("alpha", registry, router, capabilities=[])
    receiver = _make_client("beta", registry, router, capabilities=["echo"])

    async def echo_handler(msg: A2AMessage) -> dict:
        return {"echoed": msg.payload.get("text", ""), "from": msg.sender_id}

    receiver.register_handler("echo", echo_handler)

    reply = await sender.request("beta", "echo", {"text": "hello"})

    assert reply == {"echoed": "hello", "from": "alpha"}

    # Two messages flowed through: the request and the response.
    assert len(log) == 2
    assert log[0].kind == "request"
    assert log[0].sender_id == "alpha"
    assert log[1].kind == "response"
    assert log[1].correlation_id == log[0].message_id


@pytest.mark.asyncio
async def test_request_to_unknown_recipient_raises():
    registry, router, _ = _build_fabric()
    sender = _make_client("alpha", registry, router, capabilities=[])

    with pytest.raises(ValueError, match="a2a_recipient_not_registered"):
        await sender.request("nobody", "any_capability", {})


@pytest.mark.asyncio
async def test_request_to_recipient_lacking_capability_raises():
    registry, router, _ = _build_fabric()
    sender = _make_client("alpha", registry, router, capabilities=[])
    receiver = _make_client("beta", registry, router, capabilities=["other"])

    with pytest.raises(ValueError, match="does_not_advertise_capability"):
        await sender.request("beta", "missing_capability", {})


@pytest.mark.asyncio
async def test_request_without_registered_handler_raises():
    registry, router, _ = _build_fabric()
    sender = _make_client("alpha", registry, router, capabilities=[])
    receiver = _make_client("beta", registry, router, capabilities=["work"])
    # Recipient advertises 'work' but never registers a handler for it.

    with pytest.raises(ValueError, match="no_handler_registered"):
        await sender.request("beta", "work", {})


@pytest.mark.asyncio
async def test_find_capable_peers_excludes_self():
    registry, router, _ = _build_fabric()
    alpha = _make_client("alpha", registry, router, capabilities=["work"])
    beta = _make_client("beta", registry, router, capabilities=["work"])
    gamma = _make_client("gamma", registry, router, capabilities=["other"])

    peers = alpha.find_capable_peers("work")
    assert {c.agent_id for c in peers} == {"beta"}


@pytest.mark.asyncio
async def test_logger_records_every_message():
    log: list[A2AMessage] = []
    registry = A2ARegistry()
    router = A2ARouter(registry=registry, logger=log.append)
    sender = _make_client("alpha", registry, router, capabilities=[])
    receiver = _make_client("beta", registry, router, capabilities=["respond"])

    async def handler(_msg: A2AMessage) -> dict:
        return {"ok": True}

    receiver.register_handler("respond", handler)

    await sender.request("beta", "respond", {"q": "x"})
    await sender.request("beta", "respond", {"q": "y"})

    # 2 requests + 2 responses = 4 logged messages
    assert len(log) == 4
    assert sum(1 for m in log if m.kind == "request") == 2
    assert sum(1 for m in log if m.kind == "response") == 2
