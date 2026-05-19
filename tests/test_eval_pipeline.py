"""
Tests for the analysis pipeline.

Uses fabricated events.jsonl files in a tmp_path to exercise the loader,
factor extraction, comparison stats, and trajectory regression without
hitting the real Anthropic API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.compare import compare_all_factors, paired_comparison
from src.eval.efficiency import efficiency_summary, tool_calls_per_ticket
from src.eval.factors import FACTOR_COLUMNS, FactorRow, annotate_factors
from src.eval.loader import (
    discover_runs,
    load_pair,
    load_run,
    probe_interactions_df,
    session_summary_df,
)
from src.eval.trajectory import trajectories, trajectory_summary


# --- Test fixtures --------------------------------------------------

def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _make_pair(tmp_path: Path, run_id: str, pair_id: str,
               treatment_factor_at_sweep: dict[int, str] | None = None,
               control_factor_at_sweep: dict[int, str] | None = None) -> Path:
    """
    Build a (pair_id) directory with treatment + control events.jsonl.
    Each sweep has the same probe_id; the text content per sweep can
    differ so that drift trajectories have signal.
    """
    treatment_at = treatment_factor_at_sweep or {0: "calm short reply.", 1: "calm short reply."}
    control_at = control_factor_at_sweep or {0: "calm short reply.", 1: "calm short reply."}
    pair_dir = tmp_path / "runs" / run_id / pair_id
    base_ts = 100_000.0

    for arm, text_by_sweep in [("treatment", treatment_at), ("control", control_at)]:
        events: list[dict] = [
            {"ts": base_ts, "kind": "session_start", "arm": arm, "run_id": f"{run_id}:{arm}"},
        ]
        ticket_ts = base_ts + 1.0
        for sweep, text in text_by_sweep.items():
            events.append({"ts": ticket_ts, "kind": "probe_sweep_start", "arm": arm, "run_id": f"{run_id}:{arm}"})
            events.append({
                "ts": ticket_ts + 0.1,
                "kind": "probe_injected",
                "probe_id": "p_demo",
                "probe_type": "behavioral",
                "agent_id": "primary_support_01",
                "prompt": "demo probe",
            })
            events.append({
                "ts": ticket_ts + 0.2,
                "kind": "probe_response",
                "probe_id": "p_demo",
                "response": text,
                "metrics": {
                    "response_length_chars": len(text),
                    "response_length_words": len(text.split()),
                },
            })
            events.append({"ts": ticket_ts + 0.3, "kind": "probe_sweep_end", "arm": arm})
            ticket_ts += 1.0

        # Drop in a ticket with two MCP calls.
        events.append({
            "ts": ticket_ts,
            "kind": "ticket_dispatched",
            "arm": arm,
            "ticket_template_id": "demo_template",
            "customer_id": "cust_001",
        })
        events.append({"ts": ticket_ts + 0.1, "kind": "mcp_tool_call",
                       "server": "customer_db", "tool": "get_customer"})
        events.append({"ts": ticket_ts + 0.2, "kind": "mcp_tool_call",
                       "server": "knowledge_base", "tool": "search_kb"})
        events.append({"ts": ticket_ts + 0.3, "kind": "agent_response",
                       "agent_id": "primary_support_01", "response": "I'll help."})

        events.append({"ts": ticket_ts + 1.0, "kind": "session_end", "arm": arm})

        _write_events(pair_dir / arm / "events.jsonl", events)
    return pair_dir.parent  # the run dir


# --- Loader tests ---------------------------------------------------

def test_load_pair_finds_both_arms(tmp_path):
    run_dir = _make_pair(tmp_path, run_id="run_x", pair_id="pair_a")
    pair = load_pair(run_dir / "pair_a")
    assert pair.treatment is not None
    assert pair.control is not None
    assert pair.is_complete()


def test_load_run_collects_pairs(tmp_path):
    run_dir = _make_pair(tmp_path, "run_x", "pair_a")
    _make_pair(tmp_path, "run_x", "pair_b")
    run = load_run(run_dir)
    assert len(run.pairs) == 2
    assert {p.pair_id for p in run.pairs} == {"pair_a", "pair_b"}


def test_discover_runs_returns_distinct_run_dirs(tmp_path):
    _make_pair(tmp_path, "run_x", "pair_a")
    _make_pair(tmp_path, "run_y", "pair_a")
    runs = discover_runs(tmp_path / "runs")
    names = {p.name for p in runs}
    assert names == {"run_x", "run_y"}


def test_session_summary_counts_events(tmp_path):
    run_dir = _make_pair(tmp_path, "run_x", "pair_a")
    run = load_run(run_dir)
    summary = session_summary_df(run)
    assert len(summary) == 2  # treatment + control
    treatment_row = summary[summary["arm"] == "treatment"].iloc[0]
    assert treatment_row["tool_calls"] == 2
    assert treatment_row["tickets_dispatched"] == 1
    assert treatment_row["probe_responses"] >= 1


def test_probe_interactions_df_rows(tmp_path):
    run_dir = _make_pair(tmp_path, "run_x", "pair_a",
                         treatment_factor_at_sweep={0: "a", 1: "b"},
                         control_factor_at_sweep={0: "a", 1: "b"})
    run = load_run(run_dir)
    df = probe_interactions_df(run)
    # 2 arms x 2 sweeps each = 4 rows
    assert len(df) == 4
    assert set(df["arm"].unique()) == {"treatment", "control"}


# --- Factor tests ---------------------------------------------------

def test_factor_row_counts_markers():
    row = FactorRow.from_text("I am sorry but I cannot help. Perhaps you could try later. Thanks!")
    assert row.apologies >= 1
    assert row.refusal_markers >= 1
    assert row.hedges >= 1
    assert row.politeness >= 1
    assert row.sentence_count >= 2
    assert row.response_length_words > 0


def test_annotate_factors_adds_columns(tmp_path):
    run_dir = _make_pair(tmp_path, "run_x", "pair_a")
    run = load_run(run_dir)
    df = probe_interactions_df(run)
    annotated = annotate_factors(df)
    for col in FACTOR_COLUMNS:
        assert col in annotated.columns


# --- Comparison tests -----------------------------------------------

def test_paired_comparison_detects_clear_difference(tmp_path):
    # Treatment text grows longer over sweeps; control text stays short.
    run_dir = _make_pair(
        tmp_path, "run_x", "pair_a",
        treatment_factor_at_sweep={0: "short reply.", 1: "much much longer reply, several words more here."},
        control_factor_at_sweep={0: "short reply.", 1: "short reply."},
    )
    _make_pair(
        tmp_path, "run_x", "pair_b",
        treatment_factor_at_sweep={0: "short reply.", 1: "much much longer reply, several words more here."},
        control_factor_at_sweep={0: "short reply.", 1: "short reply."},
    )
    run = load_run(run_dir)
    annotated = annotate_factors(probe_interactions_df(run))
    result = paired_comparison(annotated, "response_length_words")
    # Treatment > control on response length
    assert result["mean_diff"] > 0
    assert result["n"] > 0


def test_compare_all_factors_returns_one_row_per_factor(tmp_path):
    run_dir = _make_pair(tmp_path, "run_x", "pair_a")
    run = load_run(run_dir)
    annotated = annotate_factors(probe_interactions_df(run))
    df = compare_all_factors(annotated)
    assert len(df) == len(FACTOR_COLUMNS)
    assert set(df["factor"]) == set(FACTOR_COLUMNS)


# --- Trajectory tests ----------------------------------------------

def test_trajectory_detects_increasing_slope(tmp_path):
    run_dir = _make_pair(
        tmp_path, "run_x", "pair_a",
        treatment_factor_at_sweep={0: "a", 1: "a b c d e f g h"},
        control_factor_at_sweep={0: "a", 1: "a"},
    )
    run = load_run(run_dir)
    annotated = annotate_factors(probe_interactions_df(run))
    traj = trajectories(annotated)
    treatment_slope = traj[
        (traj["arm"] == "treatment") & (traj["factor"] == "response_length_words")
    ]["slope"].iloc[0]
    assert treatment_slope > 0


def test_trajectory_summary_aggregates_by_arm_factor(tmp_path):
    run_dir = _make_pair(tmp_path, "run_x", "pair_a",
                         treatment_factor_at_sweep={0: "a", 1: "a b"},
                         control_factor_at_sweep={0: "a", 1: "a b"})
    run = load_run(run_dir)
    annotated = annotate_factors(probe_interactions_df(run))
    traj = trajectories(annotated)
    summary = trajectory_summary(traj)
    assert {"arm", "factor", "count", "mean"}.issubset(summary.columns)


# --- Efficiency tests ----------------------------------------------

def test_tool_calls_per_ticket(tmp_path):
    run_dir = _make_pair(tmp_path, "run_x", "pair_a")
    run = load_run(run_dir)
    eff = tool_calls_per_ticket(run)
    # 1 ticket per arm, 2 tool calls each.
    assert (eff["tool_calls"] == 2).all()
    assert len(eff) == 2


def test_efficiency_summary_delta(tmp_path):
    run_dir = _make_pair(tmp_path, "run_x", "pair_a")
    run = load_run(run_dir)
    eff = tool_calls_per_ticket(run)
    summary = efficiency_summary(eff)
    assert "delta_treatment_minus_control" in summary.columns
