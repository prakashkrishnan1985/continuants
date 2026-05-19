"""
Tool-call efficiency analysis.

The 1.5-2.2x event-count gap between treatment and control we noticed
mid-run is the most-visible candidate drift signal. This module turns
that observation into a real metric: tool calls per ticket, per arm,
across the session.

The hypothesis is that the treatment agent uses fewer tool calls per
ticket as the session progresses (because it has accumulated memory
and context that lets it short-circuit redundant lookups). The control
arm, which resets between tickets, should show a flat curve.
"""

from __future__ import annotations

import pandas as pd

from src.eval.loader import Run


def tool_calls_per_ticket(run: Run) -> pd.DataFrame:
    """
    For each (pair, arm, ticket_index), count the MCP tool calls
    issued between this ticket_dispatched event and the next
    ticket_dispatched event (or end of session).

    Returns a long-form DataFrame:
        run_id, pair_id, arm, ticket_index, template_id, tool_calls
    """
    rows: list[dict] = []
    for pair in run.pairs:
        for arm in (pair.treatment, pair.control):
            if arm is None:
                continue
            ticket_starts: list[tuple[int, dict]] = [
                (i, e) for i, e in enumerate(arm.events) if e.get("kind") == "ticket_dispatched"
            ]
            for ticket_idx, (event_idx, ticket_event) in enumerate(ticket_starts):
                next_event_idx = (
                    ticket_starts[ticket_idx + 1][0]
                    if ticket_idx + 1 < len(ticket_starts)
                    else len(arm.events)
                )
                slice_events = arm.events[event_idx + 1: next_event_idx]
                tool_calls = sum(1 for e in slice_events if e.get("kind") == "mcp_tool_call")
                rows.append({
                    "run_id": run.run_id,
                    "pair_id": pair.pair_id,
                    "arm": arm.arm,
                    "ticket_index": ticket_idx,
                    "template_id": ticket_event.get("ticket_template_id"),
                    "tool_calls": tool_calls,
                })
    return pd.DataFrame(rows)


def efficiency_summary(efficiency_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (pair_id, arm): mean tool calls per ticket, and (treatment minus
    control) per pair.
    """
    if efficiency_df.empty:
        return pd.DataFrame()
    by_arm = (
        efficiency_df
        .groupby(["pair_id", "arm"])["tool_calls"]
        .mean()
        .unstack("arm")
    )
    if {"treatment", "control"}.issubset(by_arm.columns):
        by_arm["delta_treatment_minus_control"] = by_arm["treatment"] - by_arm["control"]
    return by_arm.reset_index()
