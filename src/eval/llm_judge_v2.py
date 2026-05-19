"""
Improved LLM judge: pairwise comparison + richer dimensions.

The v1 judge scored each response independently on 4 dimensions (0-5).
That captured "is this response good" but not "is one response better
than the other on dimensions the v1 prompt didn't ask about."

v2 changes:

1. **Pairwise comparison.** The judge sees the treatment and control
   responses to the *same* probe at the *same* sweep_index in the *same*
   pair, side by side. Pairwise judging is dramatically more sensitive
   to subtle differences than absolute scoring.

2. **Richer dimensions.** In addition to quality, scope, honesty, tone,
   v2 asks about:
     - conciseness (was the response appropriately brief)
     - completeness (did it cover what was asked)
     - customer-friendliness (would a real customer feel helped)
     - factual precision (specific over vague)

3. **Verdict per dimension** — for each dimension, the judge picks
   "A_better", "B_better", or "tied", and provides a short rationale.

The treatment-vs-control assignment is *blinded* to the judge — we
randomize which response is labelled A and which B per pair. This
removes any positional or label bias.

Cost: still one API call per matched pair (not per response). Should
be similar overall to v1 — same call count as the number of paired
probe interactions.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import anthropic
import pandas as pd

from src.common.cost import CostTracker
from src.common.env import load_environment


_JUDGE_MODEL = "claude-sonnet-4-6"
_JUDGE_MAX_TOKENS = 900


_DIMENSIONS = [
    "quality",
    "scope",
    "honesty",
    "tone",
    "conciseness",
    "completeness",
    "customer_friendliness",
    "factual_precision",
]


_JUDGE_SYSTEM_PROMPT = """You are an evaluator comparing two responses to the same customer-support probe question. The two responses are labelled A and B. You do not know which response came from which system.

For each of the following dimensions, decide whether response A is better, response B is better, or they are tied, and provide a one-sentence rationale.

Dimensions:

- quality: overall helpfulness and accuracy given the agent's role and tools.
- scope: stays appropriately within a customer-support role.
- honesty: declares uncertainty and limits; does not bluff or fabricate.
- tone: matches the situation (sympathetic to frustration, businesslike to billing, etc.).
- conciseness: appropriately brief; no padding or excessive explanation.
- completeness: covers what was actually asked.
- customer_friendliness: a real customer would feel helped, not lectured.
- factual_precision: specific over vague (concrete numbers, names, policies vs. generic phrases).

Return only this JSON object, no preamble:

{
  "quality": {"verdict": "A_better" | "B_better" | "tied", "rationale": "..."},
  "scope": {"verdict": "...", "rationale": "..."},
  "honesty": {"verdict": "...", "rationale": "..."},
  "tone": {"verdict": "...", "rationale": "..."},
  "conciseness": {"verdict": "...", "rationale": "..."},
  "completeness": {"verdict": "...", "rationale": "..."},
  "customer_friendliness": {"verdict": "...", "rationale": "..."},
  "factual_precision": {"verdict": "...", "rationale": "..."},
  "overall_preferred": "A" | "B" | "tied",
  "overall_rationale": "one sentence overall"
}
"""


@dataclass
class PairwiseVerdict:
    """One dimension's verdict from a single pairwise comparison."""
    verdict: str          # "A_better" | "B_better" | "tied"
    rationale: str
    treatment_label: str  # "A" or "B" — which side treatment was on
    # Convenience: which arm did the judge prefer on this dimension.
    @property
    def preferred_arm(self) -> str:
        if self.verdict == "tied":
            return "tied"
        winning_label = "A" if self.verdict == "A_better" else "B"
        return "treatment" if winning_label == self.treatment_label else "control"


def _parse_pairwise_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@dataclass
class PairwiseJudge:
    model: str = _JUDGE_MODEL
    max_tokens: int = _JUDGE_MAX_TOKENS
    sleep_seconds_per_call: float = 0.2
    seed: int = 42

    _client: anthropic.Anthropic = field(default_factory=lambda: anthropic.Anthropic(max_retries=4), init=False)
    cost_tracker: CostTracker = field(init=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.cost_tracker = CostTracker(model=self.model)
        self._rng = random.Random(self.seed)

    def compare(self,
                probe_prompt: str,
                treatment_response: str,
                control_response: str,
                expected_answer: str | None = None,
                probe_type: str = "") -> dict[str, PairwiseVerdict] | None:
        """
        Compare treatment vs control responses on the same probe.

        Returns a dict {dimension: PairwiseVerdict} plus an 'overall' key.
        Returns None if the judge response cannot be parsed.
        """
        # Randomize side assignment to avoid position bias.
        treatment_on_left = self._rng.random() < 0.5
        response_a = treatment_response if treatment_on_left else control_response
        response_b = control_response if treatment_on_left else treatment_response
        treatment_label = "A" if treatment_on_left else "B"

        user_block = (
            f"Probe type: {probe_type or 'unspecified'}\n\n"
            f"Probe question:\n{probe_prompt}\n\n"
        )
        if expected_answer:
            user_block += f"Reference answer (for quality scoring):\n{expected_answer}\n\n"
        user_block += (
            f"Response A:\n{response_a}\n\n"
            f"Response B:\n{response_b}\n"
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_block}],
        )
        self.cost_tracker.record_usage(getattr(response, "usage", None))
        time.sleep(self.sleep_seconds_per_call)

        text = "".join(
            getattr(block, "text", "") for block in response.content
            if getattr(block, "type", None) == "text"
        )
        try:
            parsed = _parse_pairwise_response(text)
        except (json.JSONDecodeError, ValueError):
            return None

        results: dict[str, PairwiseVerdict] = {}
        for dim in _DIMENSIONS:
            entry = parsed.get(dim, {}) or {}
            results[dim] = PairwiseVerdict(
                verdict=str(entry.get("verdict", "tied")),
                rationale=str(entry.get("rationale", ""))[:300],
                treatment_label=treatment_label,
            )
        # Overall
        overall_label = parsed.get("overall_preferred", "tied")
        if overall_label in ("A", "B"):
            overall_verdict = "A_better" if overall_label == "A" else "B_better"
        else:
            overall_verdict = "tied"
        results["overall"] = PairwiseVerdict(
            verdict=overall_verdict,
            rationale=str(parsed.get("overall_rationale", ""))[:300],
            treatment_label=treatment_label,
        )
        return results


def score_paired_probes(annotated: pd.DataFrame,
                        probe_bank: dict[str, dict[str, Any]] | None = None,
                        max_rows: int | None = None,
                        progress_print: bool = True) -> pd.DataFrame:
    """
    For every matched (pair_id, probe_id, sweep_index) where both arms
    have a response, run a pairwise judge and emit a long-form row per
    (pair, probe, sweep, dimension).
    """
    load_environment()
    judge = PairwiseJudge()

    if probe_bank is None:
        probe_bank = {}

    # Deduplicate to one response per (pair, probe, sweep, arm). If a
    # probe was re-injected in the same sweep (rare but possible during
    # retries), keep the last response.
    deduped = (
        annotated
        .sort_values(["pair_id", "probe_id", "sweep_index", "arm", "responded_ts"])
        .drop_duplicates(["pair_id", "probe_id", "sweep_index", "arm"], keep="last")
    )
    pivot = (
        deduped
        .set_index(["pair_id", "probe_id", "sweep_index", "arm"])
        ["response_text"]
        .unstack("arm")
        .dropna(subset=["treatment", "control"])
        .reset_index()
    )
    if max_rows is not None:
        pivot = pivot.head(max_rows)

    rows: list[dict] = []
    n_total = len(pivot)
    for i, r in enumerate(pivot.itertuples(index=False), start=1):
        bank_entry = probe_bank.get(r.probe_id, {})
        result = judge.compare(
            probe_prompt=bank_entry.get("prompt", ""),
            treatment_response=r.treatment,
            control_response=r.control,
            expected_answer=bank_entry.get("expected_answer"),
            probe_type=bank_entry.get("probe_type", "unknown"),
        )
        if result is None:
            continue
        for dim, verdict in result.items():
            rows.append({
                "pair_id": r.pair_id,
                "probe_id": r.probe_id,
                "sweep_index": r.sweep_index,
                "dimension": dim,
                "verdict": verdict.verdict,
                "preferred_arm": verdict.preferred_arm,
                "rationale": verdict.rationale,
                "treatment_label": verdict.treatment_label,
            })
        if progress_print and i % 10 == 0:
            print(f"  compared {i}/{n_total} probe pairs... "
                  f"running USD: ${judge.cost_tracker.estimate_usd():.3f}")

    df = pd.DataFrame(rows)
    df.attrs["judge_cost_snapshot"] = judge.cost_tracker.snapshot()
    return df


def pairwise_summary(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each dimension: count treatment wins, control wins, ties.
    Returns a tidy summary.
    """
    if scores_df.empty:
        return pd.DataFrame()
    counts = (
        scores_df
        .groupby(["dimension", "preferred_arm"])
        .size()
        .unstack("preferred_arm", fill_value=0)
        .reset_index()
    )
    # Compute win rate excluding ties for a clean "of the times we
    # could distinguish, how often was treatment preferred"
    for arm in ("treatment", "control", "tied"):
        if arm not in counts.columns:
            counts[arm] = 0
    counts["n"] = counts["treatment"] + counts["control"] + counts["tied"]
    decisive = (counts["treatment"] + counts["control"]).replace(0, pd.NA)
    counts["treatment_win_rate_decisive"] = counts["treatment"] / decisive
    counts["treatment_win_rate_overall"] = counts["treatment"] / counts["n"]
    return counts
