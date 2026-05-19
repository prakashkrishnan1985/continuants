"""
End-to-end smoke test: live Anthropic API.

Spins up the primary support agent + technical specialist agent, wires
them together over A2A, hands the primary a single customer ticket, and
records everything to an event log.

Requires the ANTHROPIC_API_KEY environment variable to be set. This
script is intentionally NOT part of the pytest suite because it costs
real API tokens. Run it manually:

    python -m experiments.smoke_e2e.run_smoke

Output:
  experiments/smoke_e2e/runs/<timestamp>/events.jsonl
  experiments/smoke_e2e/runs/<timestamp>/summary.txt
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure project root on path when invoked directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.env import load_environment, require
from src.a2a import A2AClient, A2ARegistry, A2ARouter, AgentCard
from src.a2a.integration import register_agent_as_handler
from src.agents.primary_support import make_primary_support_agent
from src.agents.technical_specialist import make_technical_specialist_agent
from src.common.event_log import EventLog


SAMPLE_TICKET = """\
Customer: cust_001
Subject: Order ord_a1b2 arrived damaged

Body:
Hi, I received order ord_a1b2 yesterday and the wireless headphones inside
have a cracked ear cup. I have photos. I would like a replacement. The
order was delivered to my home address. Please let me know what to do.
"""


async def main() -> None:
    load_environment()
    try:
        require("ANTHROPIC_API_KEY")
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    run_dir = Path(__file__).parent / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    log = EventLog(path=run_dir / "events.jsonl", run_id=run_dir.name)
    print(f"Logging to {log.path}")

    # --- Build the A2A fabric ---------------------------------------
    registry = A2ARegistry()
    router = A2ARouter(registry=registry, logger=log.a2a_message_logger())

    primary_client = A2AClient(
        agent_id="primary_support_01", registry=registry, router=router
    )
    primary_client.register_self(AgentCard(
        agent_id="primary_support_01",
        name="Primary Customer Support Agent",
        version="0.1.0",
        capabilities=[],
        description="First-line support agent for the e-commerce store.",
    ))

    specialist_client = A2AClient(
        agent_id="tech_specialist_01", registry=registry, router=router
    )
    specialist_client.register_self(AgentCard(
        agent_id="tech_specialist_01",
        name="Technical Specialist Agent",
        version="0.1.0",
        capabilities=["diagnose_technical_issue"],
        description="Second-line technical specialist.",
    ))

    # --- Build the agents and their toolboxes ----------------------
    primary_agent, primary_toolbox = make_primary_support_agent(
        agent_id="primary_support_01",
        memory_path=run_dir / "primary_memory.json",
        logger=log.mcp_tool_logger(),
        a2a_client=primary_client,
        technical_specialist_id="tech_specialist_01",
    )

    specialist_agent, specialist_toolbox = make_technical_specialist_agent(
        agent_id="tech_specialist_01",
        memory_path=run_dir / "specialist_memory.json",
        logger=log.mcp_tool_logger(),
    )

    # --- Run the scenario ------------------------------------------
    async with primary_toolbox, specialist_toolbox:
        # Register the specialist as the A2A handler for technical diagnosis.
        register_agent_as_handler(
            client=specialist_client,
            agent=specialist_agent,
            capability="diagnose_technical_issue",
            reset_conversation_per_request=False,
        )

        log.write("scenario_start", ticket=SAMPLE_TICKET)
        print("Handing ticket to primary agent...")
        try:
            reply = await primary_agent.run_turn(SAMPLE_TICKET)
            log.log_agent_response("primary_support_01", reply)
            print("\n=== Primary agent final response ===\n")
            print(reply)
        except Exception as exc:  # noqa: BLE001
            log.log_error("primary_run_turn", str(exc))
            print(f"ERROR: {exc}")
            raise

    # --- Summary ---------------------------------------------------
    events = log.read_all()
    summary_lines = [
        f"Run id: {log.run_id}",
        f"Events recorded: {len(events)}",
        f"  MCP tool calls: {sum(1 for e in events if e['kind'] == 'mcp_tool_call')}",
        f"  A2A messages: {sum(1 for e in events if e['kind'] == 'a2a_message')}",
        f"  Agent responses: {sum(1 for e in events if e['kind'] == 'agent_response')}",
        f"  Errors: {sum(1 for e in events if e['kind'] == 'error')}",
    ]
    summary = "\n".join(summary_lines)
    (run_dir / "summary.txt").write_text(summary + "\n")
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
