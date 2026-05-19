"""
Centralized event log for an experiment run.

Writes structured events to a JSONL file (one JSON event per line) so a
later analysis pass can replay everything that happened. Provides sink
factories that plug into the existing inspectability hooks:

  - `EventLog.mcp_tool_logger()` returns a `ToolCallLogger` suitable for
    `Toolbox(server_specs=..., logger=event_log.mcp_tool_logger())`.
  - `EventLog.a2a_message_logger()` returns an `A2AMessageLogger`.
  - Direct `write(...)` for probe injections, agent responses, errors,
    or any other custom event.

Designed to be the single inspectability surface across an experiment.
This is the operational realization of point 7 of the agent definition.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.a2a.protocol import A2AMessage
from src.common.mcp_client import ToolCallLogger


@dataclass
class EventLog:
    path: Path
    run_id: str = ""

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _opened: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- Core writer ----------------------------------------------------

    def write(self, kind: str, **payload: Any) -> None:
        """
        Append one event to the log. Every event records `ts`, `kind`,
        and `run_id`; additional fields come from `payload`.
        """
        event = {
            "ts": time.time(),
            "kind": kind,
            "run_id": self.run_id,
            **payload,
        }
        line = json.dumps(event, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    # --- Sink factories -------------------------------------------------

    def mcp_tool_logger(self) -> ToolCallLogger:
        """Returns a callable suitable for passing as `Toolbox(logger=...)`."""
        def logger(server_name: str,
                   tool_name: str,
                   arguments: dict[str, Any],
                   result: Any) -> None:
            self.write(
                "mcp_tool_call",
                server=server_name,
                tool=tool_name,
                arguments=arguments,
                result=result,
            )
        return logger

    def a2a_message_logger(self):
        """Returns a callable suitable for `A2ARouter(logger=...)`."""
        def logger(message: A2AMessage) -> None:
            self.write(
                "a2a_message",
                sender=message.sender_id,
                recipient=message.recipient_id,
                msg_kind=message.kind,
                capability=message.capability,
                payload=message.payload,
                message_id=message.message_id,
                correlation_id=message.correlation_id,
                msg_ts=message.timestamp,
            )
        return logger

    # --- Convenience event types ----------------------------------------

    def log_agent_response(self, agent_id: str, response_text: str,
                           context: dict[str, Any] | None = None) -> None:
        self.write(
            "agent_response",
            agent_id=agent_id,
            response=response_text,
            context=context or {},
        )

    def log_probe_injected(self,
                            probe_id: str,
                            probe_type: str,
                            agent_id: str,
                            prompt: str,
                            metadata: dict[str, Any] | None = None) -> None:
        self.write(
            "probe_injected",
            probe_id=probe_id,
            probe_type=probe_type,
            agent_id=agent_id,
            prompt=prompt,
            metadata=metadata or {},
        )

    def log_probe_response(self,
                            probe_id: str,
                            response_text: str,
                            metrics: dict[str, Any] | None = None) -> None:
        self.write(
            "probe_response",
            probe_id=probe_id,
            response=response_text,
            metrics=metrics or {},
        )

    def log_error(self, where: str, error: str,
                  context: dict[str, Any] | None = None) -> None:
        self.write(
            "error",
            where=where,
            error=error,
            context=context or {},
        )

    # --- Readback for tests / analysis ----------------------------------

    def read_all(self) -> list[dict[str, Any]]:
        """Read the entire event log as a list of dicts (recent first not enforced)."""
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
        return events

    def count(self) -> int:
        return len(self.read_all())
