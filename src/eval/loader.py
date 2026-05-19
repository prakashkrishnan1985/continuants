"""
Load and structure event logs for analysis.

Each run produces a directory tree like:

    runs/<timestamp>/pair_<id>/treatment/events.jsonl
    runs/<timestamp>/pair_<id>/control/events.jsonl
    runs/<timestamp>/pair_<id>/treatment/primary_memory.json
    ...

This module walks a runs/ directory and yields structured records:

    Run: top-level run timestamp
      Pair: one paired (treatment+control) session
        Arm: one session ("treatment" or "control") with its events
          Events: parsed JSONL records (dicts)

It also assembles a pandas DataFrame of *probe interactions* — one row
per (probe_injected, matching probe_response) pair — annotated with
session_id, arm, pair, sweep position, and elapsed seconds. This is
the primary table the rest of the analysis pipeline operates on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd


# --- Records --------------------------------------------------------

@dataclass
class Arm:
    arm: str                       # "treatment" or "control"
    events_path: Path
    events: list[dict] = field(default_factory=list)
    memory_path: Path | None = None

    def session_age_seconds(self) -> float:
        if not self.events:
            return 0.0
        return self.events[-1]["ts"] - self.events[0]["ts"]

    def completed(self) -> bool:
        return any(e.get("kind") == "session_end" for e in self.events)

    def errors(self) -> list[dict]:
        return [e for e in self.events if e.get("kind") == "error"]


@dataclass
class Pair:
    pair_id: str
    pair_dir: Path
    treatment: Arm | None = None
    control: Arm | None = None

    def is_complete(self) -> bool:
        return (
            self.treatment is not None and self.treatment.completed()
            and self.control is not None and self.control.completed()
        )


@dataclass
class Run:
    run_id: str
    run_dir: Path
    pairs: list[Pair] = field(default_factory=list)


# --- Loading --------------------------------------------------------

def load_arm(events_path: Path, arm_name: str) -> Arm:
    arm = Arm(arm=arm_name, events_path=events_path)
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                arm.events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    mem_path = events_path.parent / ("primary_memory.json" if arm_name == "treatment" else "agent_memory.json")
    arm.memory_path = mem_path if mem_path.exists() else None
    return arm


def load_pair(pair_dir: Path) -> Pair:
    pair = Pair(pair_id=pair_dir.name, pair_dir=pair_dir)
    treatment_events = pair_dir / "treatment" / "events.jsonl"
    control_events = pair_dir / "control" / "events.jsonl"
    if treatment_events.exists():
        pair.treatment = load_arm(treatment_events, "treatment")
    if control_events.exists():
        pair.control = load_arm(control_events, "control")
    return pair


def load_run(run_dir: Path) -> Run:
    run = Run(run_id=run_dir.name, run_dir=run_dir)
    for pair_dir in sorted(run_dir.glob("pair_*")):
        if not pair_dir.is_dir():
            continue
        run.pairs.append(load_pair(pair_dir))
    return run


def discover_runs(experiments_root: Path) -> list[Path]:
    """Find every run directory under experiments/ (folders matching YYYYMMDD-HHMMSS)."""
    candidates: list[Path] = []
    if not experiments_root.exists():
        return candidates
    for path in experiments_root.rglob("events.jsonl"):
        # The run dir is the parent of pair_* parent.
        # pair_dir/<arm>/events.jsonl -> pair_dir = path.parent.parent, run_dir = pair_dir.parent
        run_dir = path.parent.parent.parent
        if run_dir not in candidates and run_dir.exists():
            candidates.append(run_dir)
    return sorted(set(candidates))


# --- Probe-interaction DataFrame ------------------------------------

def probe_interactions_df(run: Run) -> pd.DataFrame:
    """
    Build a tidy DataFrame of one row per probe interaction.

    Columns:
      run_id, pair_id, arm, probe_id, probe_type, sweep_index,
      injected_ts, responded_ts, elapsed_seconds, latency_seconds,
      response_text, response_length_chars, response_length_words
    """
    rows: list[dict] = []
    for pair in run.pairs:
        for arm in (pair.treatment, pair.control):
            if arm is None:
                continue
            session_start_ts = arm.events[0]["ts"] if arm.events else 0
            sweep_index = -1
            in_sweep = False
            # Build a map probe_id -> next-injected (queue) so we can match responses
            for event in arm.events:
                kind = event.get("kind")
                if kind in ("probe_sweep_start", "probe_sweep"):
                    sweep_index += 1
                    in_sweep = True
                elif kind == "probe_sweep_end":
                    in_sweep = False
                if kind == "probe_response":
                    metrics = event.get("metrics") or {}
                    rows.append({
                        "run_id": run.run_id,
                        "pair_id": pair.pair_id,
                        "arm": arm.arm,
                        "probe_id": event.get("probe_id"),
                        "probe_type": _infer_probe_type(arm.events, event.get("probe_id")),
                        "sweep_index": sweep_index,
                        "responded_ts": event.get("ts"),
                        "elapsed_seconds": event.get("ts", 0) - session_start_ts,
                        "response_text": event.get("response", ""),
                        "response_length_chars": metrics.get("response_length_chars", 0),
                        "response_length_words": metrics.get("response_length_words", 0),
                    })
    return pd.DataFrame(rows)


def _infer_probe_type(events: list[dict], probe_id: str) -> str:
    """Find the probe_type recorded in the matching probe_injected event."""
    if not probe_id:
        return "unknown"
    for e in events:
        if e.get("kind") == "probe_injected" and e.get("probe_id") == probe_id:
            return e.get("probe_type", "unknown")
    return "unknown"


# --- Session-level aggregates --------------------------------------

def session_summary_df(run: Run) -> pd.DataFrame:
    """One row per (pair, arm) summarizing the session."""
    rows: list[dict] = []
    for pair in run.pairs:
        for arm in (pair.treatment, pair.control):
            if arm is None:
                continue
            events = arm.events
            n = len(events)
            n_tools = sum(1 for e in events if e.get("kind") == "mcp_tool_call")
            n_a2a = sum(1 for e in events if e.get("kind") == "a2a_message")
            n_tickets = sum(1 for e in events if e.get("kind") == "ticket_dispatched")
            n_responses = sum(1 for e in events if e.get("kind") == "agent_response")
            n_probes = sum(1 for e in events if e.get("kind") == "probe_response")
            n_errors = sum(1 for e in events if e.get("kind") == "error")
            n_memory_writes = sum(
                1 for e in events
                if e.get("kind") == "mcp_tool_call" and e.get("tool") == "write_memory"
            )
            rows.append({
                "run_id": run.run_id,
                "pair_id": pair.pair_id,
                "arm": arm.arm,
                "total_events": n,
                "tool_calls": n_tools,
                "a2a_messages": n_a2a,
                "tickets_dispatched": n_tickets,
                "agent_responses": n_responses,
                "probe_responses": n_probes,
                "memory_writes": n_memory_writes,
                "errors": n_errors,
                "session_age_seconds": arm.session_age_seconds(),
                "completed": arm.completed(),
            })
    return pd.DataFrame(rows)
