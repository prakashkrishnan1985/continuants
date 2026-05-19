"""
Tests for event log and probe runner.

The probe runner test uses a fake agent that does not call Anthropic; it
just records the prompts it received and returns canned responses. This
keeps the test fast and free.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.common.event_log import EventLog
from src.probes.probe_bank import BEHAVIORAL_PROBES, STANDARD_TASK_PROBES
from src.probes.probe_types import BehavioralProbe
from src.probes.runner import ProbeRunner


@dataclass
class _StubAgent:
    agent_id: str = "stub_agent_01"
    response_template: str = "OK ({prompt_prefix})"
    seen_prompts: list[str] = field(default_factory=list)

    async def run_turn(self, prompt: str) -> str:
        self.seen_prompts.append(prompt)
        return self.response_template.format(prompt_prefix=prompt[:30])

    def reset(self) -> None:
        self.seen_prompts = []


def test_event_log_writes_and_reads_back(tmp_path):
    log = EventLog(path=tmp_path / "events.jsonl", run_id="test_run_001")
    log.write("custom_event", note="hello", value=42)
    log.log_agent_response("agent_x", "The agent said this.")
    log.log_error("nowhere", "boom")

    events = log.read_all()
    assert len(events) == 3
    assert events[0]["kind"] == "custom_event"
    assert events[0]["note"] == "hello"
    assert events[0]["value"] == 42
    assert events[0]["run_id"] == "test_run_001"
    assert events[1]["kind"] == "agent_response"
    assert events[2]["kind"] == "error"


def test_event_log_mcp_logger_records_tool_calls(tmp_path):
    log = EventLog(path=tmp_path / "events.jsonl")
    mcp_log = log.mcp_tool_logger()
    mcp_log("customer_db", "get_customer", {"customer_id": "cust_001"}, {"name": "Alex"})
    mcp_log("memory", "write_memory", {"body": "note"}, "mem_xyz")

    events = log.read_all()
    assert len(events) == 2
    assert events[0]["kind"] == "mcp_tool_call"
    assert events[0]["server"] == "customer_db"
    assert events[0]["tool"] == "get_customer"
    assert events[0]["arguments"] == {"customer_id": "cust_001"}
    assert events[0]["result"] == {"name": "Alex"}


def test_event_log_a2a_logger_records_messages(tmp_path):
    from src.a2a.protocol import A2AMessage

    log = EventLog(path=tmp_path / "events.jsonl")
    a2a_log = log.a2a_message_logger()
    msg = A2AMessage(
        sender_id="primary",
        recipient_id="specialist",
        kind="request",
        capability="diagnose",
        payload={"q": "?"},
    )
    a2a_log(msg)

    events = log.read_all()
    assert len(events) == 1
    assert events[0]["kind"] == "a2a_message"
    assert events[0]["sender"] == "primary"
    assert events[0]["recipient"] == "specialist"
    assert events[0]["msg_kind"] == "request"


@pytest.mark.asyncio
async def test_probe_runner_inject_one_logs_prompt_and_response(tmp_path):
    log = EventLog(path=tmp_path / "events.jsonl", run_id="probe_test_001")
    agent = _StubAgent(agent_id="primary_001", response_template="reply for: {prompt_prefix}")
    probe = BehavioralProbe(probe_id="bh_test", prompt="Say hello to the customer.")
    runner = ProbeRunner(agent=agent, probes=[probe], event_log=log)

    response = await runner.inject_one(probe)
    assert response.startswith("reply for:")

    events = log.read_all()
    assert len(events) == 2
    assert events[0]["kind"] == "probe_injected"
    assert events[0]["probe_id"] == "bh_test"
    assert events[0]["probe_type"] == "behavioral"
    assert events[0]["agent_id"] == "primary_001"
    assert events[1]["kind"] == "probe_response"
    assert events[1]["probe_id"] == "bh_test"
    assert events[1]["metrics"]["response_length_words"] > 0


@pytest.mark.asyncio
async def test_probe_runner_inject_all_runs_every_probe_in_order(tmp_path):
    log = EventLog(path=tmp_path / "events.jsonl")
    agent = _StubAgent()
    probes = list(BEHAVIORAL_PROBES[:3])
    runner = ProbeRunner(agent=agent, probes=probes, event_log=log)

    responses = await runner.inject_all()
    assert len(responses) == 3
    assert agent.seen_prompts == [p.prompt for p in probes]

    events = log.read_all()
    inject_events = [e for e in events if e["kind"] == "probe_injected"]
    assert [e["probe_id"] for e in inject_events] == [p.probe_id for p in probes]


@pytest.mark.asyncio
async def test_probe_runner_maybe_inject_interval(tmp_path, monkeypatch):
    log = EventLog(path=tmp_path / "events.jsonl")
    agent = _StubAgent()
    probes = list(STANDARD_TASK_PROBES[:2])
    runner = ProbeRunner(agent=agent, probes=probes, event_log=log)

    # Patch elapsed_seconds to control time deterministically.
    elapsed_ref = [0.0]
    monkeypatch.setattr(runner, "elapsed_seconds", lambda: elapsed_ref[0])

    # No interval elapsed yet.
    fired = await runner.maybe_inject_interval(interval_seconds=60.0)
    assert fired is False

    # Cross the first interval boundary.
    elapsed_ref[0] = 65.0
    fired = await runner.maybe_inject_interval(interval_seconds=60.0)
    assert fired is True

    # Same interval, calling again should not re-fire.
    elapsed_ref[0] = 90.0
    fired = await runner.maybe_inject_interval(interval_seconds=60.0)
    assert fired is False

    # Cross the second interval boundary.
    elapsed_ref[0] = 125.0
    fired = await runner.maybe_inject_interval(interval_seconds=60.0)
    assert fired is True

    # We expect two full sweeps through the probe list, i.e., 2 * len(probes).
    events = log.read_all()
    inject_events = [e for e in events if e["kind"] == "probe_injected"]
    assert len(inject_events) == 2 * len(probes)
