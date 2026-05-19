"""
Probe runner.

Coordinates injecting probes into an agent's session at controlled
time points and recording the agent's response. Designed to be driven
either by elapsed-time (every N seconds since session start) or by
explicit caller invocation between regular agent turns.

Critical design choice: the runner does not score probe responses
live. Scoring happens in the analysis pass against the recorded event
log. This keeps the runner small and ensures every scoring rule can be
applied to every prior run, including retroactively.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable

from src.agents.base_agent import BaseAgent
from src.common.event_log import EventLog
from src.probes.probe_types import Probe


@dataclass
class ProbeRunner:
    agent: BaseAgent
    probes: list[Probe]
    event_log: EventLog
    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:8]}")

    _session_start: float = field(default_factory=time.monotonic, init=False, repr=False)
    _intervals_fired: set[int] = field(default_factory=set, init=False, repr=False)

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._session_start

    async def inject_one(self, probe: Probe) -> str:
        """
        Inject a single probe. Logs the prompt, runs it through the
        agent's current conversation (the conversation is not reset),
        logs the response. Returns the response text.

        Side effect: the probe becomes part of the agent's conversation
        history. That is deliberate. Probes are part of the session;
        they cannot be hidden from the agent without breaking the
        treatment-vs-control comparison.
        """
        self.event_log.log_probe_injected(
            probe_id=probe.probe_id,
            probe_type=probe.probe_type,
            agent_id=self.agent.agent_id,
            prompt=probe.prompt,
            metadata={
                "session_id": self.session_id,
                "elapsed_seconds": self.elapsed_seconds(),
                **probe.metadata,
            },
        )
        try:
            response = await self.agent.run_turn(probe.prompt)
        except Exception as exc:  # noqa: BLE001
            self.event_log.log_error(
                where=f"probe_runner:{probe.probe_id}",
                error=str(exc),
                context={"session_id": self.session_id},
            )
            raise

        self.event_log.log_probe_response(
            probe_id=probe.probe_id,
            response_text=response,
            metrics={
                "session_id": self.session_id,
                "elapsed_seconds": self.elapsed_seconds(),
                "response_length_chars": len(response),
                "response_length_words": len(response.split()),
                "agent_id": self.agent.agent_id,
            },
        )
        return response

    async def inject_all(self, probes: Iterable[Probe] | None = None) -> list[str]:
        """Inject every probe in sequence. Returns ordered responses."""
        targets = list(probes if probes is not None else self.probes)
        responses: list[str] = []
        for probe in targets:
            responses.append(await self.inject_one(probe))
        return responses

    async def maybe_inject_interval(self,
                                    interval_seconds: float,
                                    probes: Iterable[Probe] | None = None) -> bool:
        """
        If at least `interval_seconds` have passed since the last
        interval fire (or session start), inject the probe bank and
        return True. Otherwise return False without injecting.

        Use this in an outer loop that processes real tickets, calling
        `maybe_inject_interval` between tickets. The runner tracks how
        many interval boundaries have already fired.
        """
        elapsed = self.elapsed_seconds()
        interval_idx = int(elapsed // interval_seconds)
        if interval_idx == 0:
            return False
        if interval_idx in self._intervals_fired:
            return False
        self._intervals_fired.add(interval_idx)
        await self.inject_all(probes)
        return True

    def reset_session(self) -> None:
        """Reset the elapsed-time clock and clear interval bookkeeping."""
        self._session_start = time.monotonic()
        self._intervals_fired.clear()
