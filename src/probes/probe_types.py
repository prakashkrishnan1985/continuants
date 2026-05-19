"""
Probe type definitions.

A probe is a deliberate, controlled prompt injected into a long-running
agent session to measure the agent's behaviour at that moment. Probes
come in distinct types so the analysis pass can stratify results by
what each probe was testing.

Probe types defined here:

- BehavioralProbe: a neutral prompt designed to elicit observable
  behavioural factors (response length, tone, formality, tool
  selection patterns, refusal rate). Multiple of these injected over
  time give the multi-axis fingerprint trajectory.

- StandardTaskProbe: a small fixed task with a known correct answer
  or scorable rubric. Repeated injections measure the agent's
  outcome-quality trajectory (positive vs negative drift on the
  primary task dimension).

- AdjacentDomainProbe: a task in an adjacent but distinct domain
  from the agent's primary purpose. Measures tangential drift, the
  (0,0,1) vector in our framing.

- PerturbationProbe: a deliberately constraint-violating or
  contradiction-inducing prompt. Measures how the agent's response
  to surprise changes over time (plan-deviation and self-correction
  factors).

- ReflectionProbe: an open-ended self-state question (point 4
  metacognition). The agent describes its own state, history, and
  uncertainty. Used for both metacognition measurement and
  tangential-drift detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(kw_only=True)
class Probe:
    """Base probe definition."""
    probe_id: str
    probe_type: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class BehavioralProbe(Probe):
    probe_type: str = "behavioral"


@dataclass(kw_only=True)
class StandardTaskProbe(Probe):
    """A fixed scorable task. `expected_answer` is optional ground truth
    used by the analysis pass; the probe runner does not score live."""
    probe_type: str = "standard_task"
    expected_answer: str | None = None


@dataclass(kw_only=True)
class AdjacentDomainProbe(Probe):
    probe_type: str = "adjacent_domain"


@dataclass(kw_only=True)
class PerturbationProbe(Probe):
    """A perturbation prompt. `perturbation_kind` is one of:
    contradiction, ambiguity, resource_constraint, fake_error, role_shift."""
    probe_type: str = "perturbation"
    perturbation_kind: str = "contradiction"


@dataclass(kw_only=True)
class ReflectionProbe(Probe):
    probe_type: str = "reflection"
