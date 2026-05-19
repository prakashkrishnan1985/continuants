"""
Compute behavioural factors from probe-response text.

These are the operational axes we declared in the paper: response length,
formality markers, refusal/escalation rate, self-correction frequency,
plus a few rate-limit-awareness markers.

This module is intentionally simple — every factor is a small text-pattern
counter. The point is to keep the measurement easy to inspect rather
than maximally accurate; reviewers can drop in better measurement code
without affecting the rest of the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


# --- Marker lexicons -----------------------------------------------

_HEDGE_TERMS = {
    "perhaps", "maybe", "might", "could", "seems", "i think", "possibly",
    "appears", "likely", "probably", "i'm not sure", "i am not sure",
}

_APOLOGY_TERMS = {
    "sorry", "apologize", "apologies", "regret", "my mistake",
    "i apologise", "i apologize",
}

_POLITENESS_TERMS = {
    "please", "thank you", "thanks", "appreciate", "kindly",
}

_REFUSAL_TERMS = {
    "i can't", "i cannot", "i'm unable", "i am unable",
    "i don't have", "i do not have", "outside my scope",
    "not authorized", "not authorised", "not within",
    "above my pay grade", "out of scope", "i'm sorry, i",
}

_ESCALATION_TERMS = {
    "escalate", "specialist", "supervisor", "transfer you",
    "route this", "hand this off", "pass this", "loop in",
}

_SELF_CORRECTION_TERMS = {
    "wait", "let me reconsider", "actually", "on second thought",
    "i should clarify", "correction", "to be precise", "let me revise",
}

_RATE_LIMIT_AWARENESS_TERMS = {
    "rate limit", "rate-limit", "be conservative", "budget", "tool call",
    "save calls", "minimise", "minimize", "fewer", "skip", "defer",
}


def _count_terms(text: str, terms: Iterable[str]) -> int:
    if not text:
        return 0
    lower = text.lower()
    return sum(lower.count(term) for term in terms)


_SENTENCE_RE = re.compile(r"[.!?]+")


def sentence_count(text: str) -> int:
    if not text:
        return 0
    parts = [p for p in _SENTENCE_RE.split(text) if p.strip()]
    return len(parts)


# --- Per-response factor row ---------------------------------------

@dataclass
class FactorRow:
    response_length_words: int
    response_length_chars: int
    sentence_count: int
    hedges: int
    apologies: int
    politeness: int
    refusal_markers: int
    escalation_markers: int
    self_correction_markers: int
    rate_limit_awareness_markers: int

    @classmethod
    def from_text(cls, text: str) -> "FactorRow":
        return cls(
            response_length_words=len((text or "").split()),
            response_length_chars=len(text or ""),
            sentence_count=sentence_count(text or ""),
            hedges=_count_terms(text, _HEDGE_TERMS),
            apologies=_count_terms(text, _APOLOGY_TERMS),
            politeness=_count_terms(text, _POLITENESS_TERMS),
            refusal_markers=_count_terms(text, _REFUSAL_TERMS),
            escalation_markers=_count_terms(text, _ESCALATION_TERMS),
            self_correction_markers=_count_terms(text, _SELF_CORRECTION_TERMS),
            rate_limit_awareness_markers=_count_terms(text, _RATE_LIMIT_AWARENESS_TERMS),
        )


# --- Apply to a probe-interactions DataFrame -----------------------

FACTOR_COLUMNS = [
    "response_length_words",
    "response_length_chars",
    "sentence_count",
    "hedges",
    "apologies",
    "politeness",
    "refusal_markers",
    "escalation_markers",
    "self_correction_markers",
    "rate_limit_awareness_markers",
]


def annotate_factors(probe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `probe_df` with one column per behavioural factor
    computed from the `response_text` of each row.

    If `probe_df` already has columns named identically to factor outputs
    (e.g., `response_length_words` is also written by the loader from the
    event-log metrics), the factor-derived value wins. This keeps the
    canonical factor source in one place (this module) and avoids the
    duplicate-column footgun that pandas concat silently produces.
    """
    if probe_df.empty:
        return probe_df.copy()
    rows = [FactorRow.from_text(t) for t in probe_df["response_text"].astype(str)]
    factor_df = pd.DataFrame([r.__dict__ for r in rows])
    base = probe_df.reset_index(drop=True).drop(
        columns=[c for c in FACTOR_COLUMNS if c in probe_df.columns],
        errors="ignore",
    )
    return pd.concat([base, factor_df], axis=1)
