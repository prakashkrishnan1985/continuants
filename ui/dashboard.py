"""
Live event-log dashboard for the Continuants drift experiments.

Streamlit app that tails the JSONL event log of any in-progress or
completed run and renders the events in a readable form. Auto-refreshes
every 2 seconds so you can watch a session as it happens.

Run with:

    streamlit run ui/dashboard.py

By default it scans `experiments/<study>/runs/<timestamp>/<pair>/` for
event logs. Pick which run to view from the sidebar.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- Helpers ---------------------------------------------------------

def find_event_logs() -> list[Path]:
    """Find every events.jsonl under experiments/.

    Sorted by modification time, newest first.
    """
    root = PROJECT_ROOT / "experiments"
    if not root.exists():
        return []
    candidates = list(root.rglob("events.jsonl"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def fmt_time(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def event_color(kind: str) -> str:
    return {
        "agent_response": "#1e88e5",
        "mcp_tool_call": "#7e57c2",
        "a2a_message": "#43a047",
        "probe_injected": "#fb8c00",
        "probe_response": "#ef6c00",
        "ticket_dispatched": "#546e7a",
        "session_start": "#00897b",
        "session_end": "#00695c",
        "session_aborted": "#c62828",
        "error": "#c62828",
    }.get(kind, "#90a4ae")


def render_event(event: dict) -> None:
    kind = event.get("kind", "unknown")
    ts = event.get("ts", 0)
    color = event_color(kind)

    header = f":small[`{fmt_time(ts)}`]  **:{kind_to_streamlit_color(color)}[{kind}]**"

    if kind == "ticket_dispatched":
        st.markdown(
            f"{header}  ·  `{event.get('arm', '?')}`  ·  template "
            f"`{event.get('ticket_template_id', '?')}`  ·  customer "
            f"`{event.get('customer_id', '?')}`  ·  tone "
            f"`{event.get('tone_variant', '?')}`"
        )
    elif kind == "mcp_tool_call":
        st.markdown(
            f"{header}  ·  `{event.get('server', '?')}.{event.get('tool', '?')}`"
        )
        with st.expander("args / result", expanded=False):
            st.json({"arguments": event.get("arguments"), "result": event.get("result")})
    elif kind == "a2a_message":
        st.markdown(
            f"{header}  ·  `{event.get('msg_kind', '?')}`  ·  "
            f"{event.get('sender', '?')} → {event.get('recipient', '?')}  ·  "
            f"capability `{event.get('capability', '?')}`"
        )
        with st.expander("payload", expanded=False):
            st.json(event.get("payload"))
    elif kind == "probe_injected":
        st.markdown(
            f"{header}  ·  `{event.get('probe_type', '?')}`  ·  "
            f"id `{event.get('probe_id', '?')}`"
        )
        st.caption(event.get("prompt", ""))
    elif kind == "probe_response":
        metrics = event.get("metrics", {}) or {}
        st.markdown(
            f"{header}  ·  id `{event.get('probe_id', '?')}`  ·  "
            f"{metrics.get('response_length_words', 0)} words"
        )
        with st.expander("response", expanded=False):
            st.write(event.get("response", ""))
    elif kind == "agent_response":
        st.markdown(
            f"{header}  ·  `{event.get('agent_id', '?')}`"
        )
        with st.expander("response text", expanded=False):
            st.write(event.get("response", ""))
    elif kind == "error":
        st.markdown(
            f"{header}  ·  in `{event.get('where', '?')}`"
        )
        st.error(event.get("error", ""))
    else:
        st.markdown(header)
        st.json({k: v for k, v in event.items() if k not in ("kind", "ts")})


def kind_to_streamlit_color(hex_color: str) -> str:
    """Map our colour hexes to one of streamlit's named markdown colours."""
    return {
        "#1e88e5": "blue",
        "#7e57c2": "violet",
        "#43a047": "green",
        "#fb8c00": "orange",
        "#ef6c00": "orange",
        "#546e7a": "gray",
        "#00897b": "green",
        "#00695c": "green",
        "#c62828": "red",
        "#90a4ae": "gray",
    }.get(hex_color, "gray")


# --- Page ------------------------------------------------------------

st.set_page_config(
    page_title="Continuants — Live Run Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Continuants live dashboard")
st.caption("Reads the JSONL event log of a drift-study run and renders it as it grows.")

with st.sidebar:
    st.header("Run")
    logs = find_event_logs()
    if not logs:
        st.warning("No event logs found yet under experiments/. Start a run to see data.")
        st.stop()

    log_labels = [str(p.relative_to(PROJECT_ROOT)) for p in logs]
    selected_idx = st.selectbox(
        "Event log",
        options=list(range(len(logs))),
        format_func=lambda i: log_labels[i],
    )
    selected_path = logs[selected_idx]

    auto_refresh = st.checkbox("Auto refresh", value=True)
    refresh_seconds = st.slider("Refresh seconds", 1, 10, 2)

    st.divider()
    st.header("Filters")

events = load_events(selected_path)
all_kinds = sorted({e.get("kind", "unknown") for e in events})

with st.sidebar:
    kind_filter = st.multiselect(
        "Show event kinds",
        options=all_kinds,
        default=all_kinds,
    )
    arm_filter = st.multiselect(
        "Arm",
        options=sorted({e.get("arm") for e in events if e.get("arm")}),
        default=sorted({e.get("arm") for e in events if e.get("arm")}),
    )
    max_events = st.slider("Most recent N events", 50, 5000, 500, step=50)

# --- Summary metrics -------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Events", len(events))
col2.metric("MCP calls", sum(1 for e in events if e.get("kind") == "mcp_tool_call"))
col3.metric("A2A msgs", sum(1 for e in events if e.get("kind") == "a2a_message"))
col4.metric("Probes injected", sum(1 for e in events if e.get("kind") == "probe_injected"))
col5.metric("Errors", sum(1 for e in events if e.get("kind") == "error"))

# --- Filtered timeline ----------------------------------------------

st.subheader("Event timeline")

filtered = [
    e for e in events
    if e.get("kind") in kind_filter
    and (not arm_filter or e.get("arm") in arm_filter or e.get("arm") is None)
]
filtered = filtered[-max_events:]

if not filtered:
    st.info("No events match the current filters.")
else:
    for event in reversed(filtered):  # newest first
        with st.container(border=True):
            render_event(event)

# --- Auto refresh ----------------------------------------------------

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
