"""
Experiment runner.

Orchestrates a single experimental session: build the agents, stream
tickets through them, inject probes at the configured interval, and
record everything to a single event log.

The runner has two arm types:

  - "treatment" arm: the primary support agent with persistent memory,
    accumulated conversation context across tickets, and A2A escalation
    to the technical specialist. This is the "agent" by our definition.

  - "control" arm: the workflow control variant, with no-op memory, no
    A2A escalation, and the conversation reset before every ticket.
    This is the "workflow" by our definition.

A single `run_arm(...)` call processes one full session for one arm.
Multiple sessions are needed for statistical power; the runner is the
inner loop that handles one session, and a higher-level driver iterates
it across (arm, seed) combinations.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.a2a import A2AClient, A2ARegistry, A2ARouter, AgentCard
from src.a2a.integration import register_agent_as_handler
from src.agents.base_agent import BaseAgent
from src.agents.primary_support import make_primary_support_agent
from src.agents.technical_specialist import make_technical_specialist_agent
from src.agents.workflow_control import make_workflow_control_agent
from src.common.event_log import EventLog
from src.common.mcp_client import Toolbox
from src.probes.probe_bank import DEFAULT_PROBE_BANK
from src.probes.runner import ProbeRunner
from src.probes.ticket_generator import Ticket
from src.probes.ticket_source import build_ticket_source


# --- Configuration ---------------------------------------------------

@dataclass
class ArmConfig:
    """Configuration for a single arm of a single session."""

    arm: str                                # "treatment" or "control"
    run_id: str
    output_dir: Path
    n_tickets: int = 30
    probe_interval_seconds: float = 1800.0  # 30 min default
    ticket_seed: int = 42
    model: str = "claude-sonnet-4-6"
    inject_initial_probes: bool = True
    inject_final_probes: bool = True
    max_consecutive_ticket_errors: int = 3
    # Per-arm budget cap in USD. If the cumulative estimated spend for
    # this single arm exceeds this number, the session aborts gracefully.
    # 0.0 disables budget enforcement.
    max_budget_usd: float = 0.0
    # Which ticket stream to use. "templated" is the synthetic in-code
    # generator; "bitext" loads real customer-support data from a
    # local CSV. See src/probes/ticket_source.py.
    ticket_source: str = "templated"


# --- Helpers ---------------------------------------------------------

def _format_ticket_for_agent(ticket: Ticket) -> str:
    """Render a ticket into the prompt the agent receives."""
    lines = [
        f"New support ticket from customer {ticket.customer_id}.",
        f"Subject: {ticket.subject}",
        f"Priority: {ticket.priority}",
    ]
    if ticket.related_order_id:
        lines.append(f"Related order: {ticket.related_order_id}")
    lines.append("")
    lines.append("Body:")
    lines.append(ticket.body)
    return "\n".join(lines)


# --- Per-arm run loops -----------------------------------------------

async def run_treatment_arm(config: ArmConfig) -> dict[str, Any]:
    """
    Run one session of the treatment arm.

    Returns a summary dict. Full event trace lives in the event log.
    """
    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    event_log = EventLog(
        path=run_dir / "events.jsonl",
        run_id=f"{config.run_id}:treatment",
    )
    event_log.write("session_start", arm="treatment", config=_serialize_config(config))

    # --- A2A fabric ----------------------------------------------------
    registry = A2ARegistry()
    router = A2ARouter(registry=registry, logger=event_log.a2a_message_logger())

    primary_client = A2AClient(
        agent_id="primary_support_01", registry=registry, router=router
    )
    primary_client.register_self(AgentCard(
        agent_id="primary_support_01",
        name="Primary Customer Support Agent",
        version="0.1.0",
        capabilities=[],
        description="First-line e-commerce support.",
    ))

    specialist_client = A2AClient(
        agent_id="tech_specialist_01", registry=registry, router=router
    )
    specialist_client.register_self(AgentCard(
        agent_id="tech_specialist_01",
        name="Technical Specialist",
        version="0.1.0",
        capabilities=["diagnose_technical_issue"],
        description="Second-line technical specialist.",
    ))

    # --- Build agents --------------------------------------------------
    primary_agent, primary_toolbox = make_primary_support_agent(
        agent_id="primary_support_01",
        memory_path=run_dir / "primary_memory.json",
        logger=event_log.mcp_tool_logger(),
        a2a_client=primary_client,
        technical_specialist_id="tech_specialist_01",
        model=config.model,
    )
    specialist_agent, specialist_toolbox = make_technical_specialist_agent(
        agent_id="tech_specialist_01",
        memory_path=run_dir / "specialist_memory.json",
        logger=event_log.mcp_tool_logger(),
        model=config.model,
    )

    summary: dict[str, Any] = {}
    async with primary_toolbox, specialist_toolbox:
        register_agent_as_handler(
            client=specialist_client,
            agent=specialist_agent,
            capability="diagnose_technical_issue",
            reset_conversation_per_request=False,
        )
        summary = await _run_session(
            arm_name="treatment",
            primary_agent=primary_agent,
            reset_between_tickets=False,
            config=config,
            event_log=event_log,
        )

    event_log.write("session_end", arm="treatment", summary=summary)
    return summary


async def run_control_arm(config: ArmConfig) -> dict[str, Any]:
    """
    Run one session of the control (workflow) arm.

    Same input stream as the treatment arm given the same `ticket_seed`,
    but no memory, no A2A, conversation reset between every ticket.
    """
    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    event_log = EventLog(
        path=run_dir / "events.jsonl",
        run_id=f"{config.run_id}:control",
    )
    event_log.write("session_start", arm="control", config=_serialize_config(config))

    agent, toolbox = make_workflow_control_agent(
        agent_id="workflow_control_01",
        logger=event_log.mcp_tool_logger(),
        model=config.model,
    )

    summary: dict[str, Any] = {}
    async with toolbox:
        summary = await _run_session(
            arm_name="control",
            primary_agent=agent,
            reset_between_tickets=True,
            config=config,
            event_log=event_log,
        )

    event_log.write("session_end", arm="control", summary=summary)
    return summary


# --- Inner session loop (shared) -------------------------------------

async def _run_session(arm_name: str,
                       primary_agent: BaseAgent,
                       reset_between_tickets: bool,
                       config: ArmConfig,
                       event_log: EventLog) -> dict[str, Any]:
    """Common session loop used by both arms."""
    ticket_gen = build_ticket_source(config.ticket_source, seed=config.ticket_seed)
    probe_runner = ProbeRunner(
        agent=primary_agent,
        probes=DEFAULT_PROBE_BANK,
        event_log=event_log,
    )

    counts = {
        "tickets_attempted": 0,
        "tickets_succeeded": 0,
        "tickets_failed": 0,
        "probe_sweeps": 0,
        "cost_snapshot": None,
    }
    consecutive_errors = 0

    # Initial probe sweep at t=0 to establish baseline fingerprint.
    if config.inject_initial_probes:
        event_log.write("probe_sweep_start", arm=arm_name, position="initial")
        await probe_runner.inject_all()
        counts["probe_sweeps"] += 1
        event_log.write("probe_sweep_end", arm=arm_name, position="initial")
        if reset_between_tickets:
            primary_agent.reset()

    # Stream tickets, injecting probes at intervals.
    for ticket in ticket_gen.generate(config.n_tickets):
        # Budget check before each new ticket. Cheap, advisory.
        if config.max_budget_usd > 0:
            spent = primary_agent.cost_tracker.estimate_usd()
            if spent >= config.max_budget_usd:
                event_log.write(
                    "session_aborted",
                    arm=arm_name,
                    reason="max_budget_reached",
                    estimated_usd=round(spent, 4),
                    budget_usd=config.max_budget_usd,
                )
                break
        counts["tickets_attempted"] += 1

        # Maybe inject probes between tickets.
        if await probe_runner.maybe_inject_interval(config.probe_interval_seconds):
            counts["probe_sweeps"] += 1
            event_log.write("probe_sweep", arm=arm_name, position="interval")
            if reset_between_tickets:
                primary_agent.reset()

        # Workflow control resets conversation before every ticket.
        if reset_between_tickets:
            primary_agent.reset()

        event_log.write(
            "ticket_dispatched",
            arm=arm_name,
            ticket_template_id=ticket.ticket_template_id,
            customer_id=ticket.customer_id,
            related_order_id=ticket.related_order_id,
            tone_variant=ticket.tone_variant,
            priority=ticket.priority,
        )

        try:
            response = await primary_agent.run_turn(_format_ticket_for_agent(ticket))
            event_log.log_agent_response(
                primary_agent.agent_id,
                response,
                context={
                    "arm": arm_name,
                    "ticket_template_id": ticket.ticket_template_id,
                    "customer_id": ticket.customer_id,
                },
            )
            counts["tickets_succeeded"] += 1
            consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001
            event_log.log_error(
                where=f"ticket:{ticket.ticket_template_id}",
                error=str(exc),
                context={"arm": arm_name},
            )
            counts["tickets_failed"] += 1
            consecutive_errors += 1
            if consecutive_errors >= config.max_consecutive_ticket_errors:
                event_log.write(
                    "session_aborted",
                    arm=arm_name,
                    reason="too_many_consecutive_ticket_errors",
                )
                break

    # Final probe sweep to capture end-of-session fingerprint.
    if config.inject_final_probes:
        if reset_between_tickets:
            primary_agent.reset()
        event_log.write("probe_sweep_start", arm=arm_name, position="final")
        await probe_runner.inject_all()
        counts["probe_sweeps"] += 1
        event_log.write("probe_sweep_end", arm=arm_name, position="final")

    counts["cost_snapshot"] = primary_agent.cost_tracker.snapshot()
    event_log.write("cost_snapshot", arm=arm_name, **counts["cost_snapshot"])
    return counts


def _serialize_config(config: ArmConfig) -> dict[str, Any]:
    return {
        "arm": config.arm,
        "run_id": config.run_id,
        "n_tickets": config.n_tickets,
        "probe_interval_seconds": config.probe_interval_seconds,
        "ticket_seed": config.ticket_seed,
        "model": config.model,
        "inject_initial_probes": config.inject_initial_probes,
        "inject_final_probes": config.inject_final_probes,
    }


# --- High-level driver (paired treatment + control) ------------------

@dataclass
class PairedSessionConfig:
    """
    Configuration for a paired run: one treatment session and one
    control session, both seeded identically so they see the same ticket
    stream.
    """

    output_dir: Path
    n_tickets: int = 30
    probe_interval_seconds: float = 1800.0
    seed: int = 42
    model: str = "claude-sonnet-4-6"
    # USD cap applied per arm (treatment and control each). 0.0 disables.
    max_budget_usd_per_arm: float = 0.0
    # See ArmConfig.ticket_source for valid values.
    ticket_source: str = "templated"


async def run_paired_session(config: PairedSessionConfig) -> dict[str, Any]:
    """Run a treatment + control pair in sequence."""
    run_id = f"pair_{uuid.uuid4().hex[:8]}"

    treatment_config = ArmConfig(
        arm="treatment",
        run_id=f"{run_id}/treatment",
        output_dir=config.output_dir,
        n_tickets=config.n_tickets,
        probe_interval_seconds=config.probe_interval_seconds,
        ticket_seed=config.seed,
        model=config.model,
        max_budget_usd=config.max_budget_usd_per_arm,
        ticket_source=config.ticket_source,
    )
    control_config = ArmConfig(
        arm="control",
        run_id=f"{run_id}/control",
        output_dir=config.output_dir,
        n_tickets=config.n_tickets,
        probe_interval_seconds=config.probe_interval_seconds,
        ticket_seed=config.seed,
        model=config.model,
        max_budget_usd=config.max_budget_usd_per_arm,
        ticket_source=config.ticket_source,
    )

    started = time.time()
    treatment_summary = await run_treatment_arm(treatment_config)
    control_summary = await run_control_arm(control_config)
    elapsed = time.time() - started

    return {
        "run_id": run_id,
        "seconds_elapsed": elapsed,
        "treatment": treatment_summary,
        "control": control_summary,
    }
