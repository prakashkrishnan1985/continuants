"""Tests for the v2 pairwise judge parsing and aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eval.llm_judge_v2 import (
    PairwiseVerdict,
    _parse_pairwise_response,
    pairwise_summary,
)


def test_parse_full_pairwise_response():
    text = """{
  "quality": {"verdict": "A_better", "rationale": "concrete"},
  "scope": {"verdict": "tied", "rationale": "both ok"},
  "honesty": {"verdict": "B_better", "rationale": "b more uncertain"},
  "tone": {"verdict": "A_better", "rationale": "a polite"},
  "conciseness": {"verdict": "B_better", "rationale": "b shorter"},
  "completeness": {"verdict": "tied", "rationale": "both covered"},
  "customer_friendliness": {"verdict": "A_better", "rationale": "a warmer"},
  "factual_precision": {"verdict": "A_better", "rationale": "a cites policy"},
  "overall_preferred": "A",
  "overall_rationale": "A is better overall"
}"""
    parsed = _parse_pairwise_response(text)
    assert parsed["quality"]["verdict"] == "A_better"
    assert parsed["overall_preferred"] == "A"


def test_pairwise_verdict_preferred_arm_when_treatment_is_A():
    v = PairwiseVerdict(verdict="A_better", rationale="", treatment_label="A")
    assert v.preferred_arm == "treatment"


def test_pairwise_verdict_preferred_arm_when_treatment_is_B():
    v = PairwiseVerdict(verdict="A_better", rationale="", treatment_label="B")
    assert v.preferred_arm == "control"


def test_pairwise_verdict_tied():
    v = PairwiseVerdict(verdict="tied", rationale="", treatment_label="A")
    assert v.preferred_arm == "tied"


def test_pairwise_summary_counts_wins():
    df = pd.DataFrame([
        {"dimension": "quality", "preferred_arm": "treatment"},
        {"dimension": "quality", "preferred_arm": "treatment"},
        {"dimension": "quality", "preferred_arm": "control"},
        {"dimension": "quality", "preferred_arm": "tied"},
        {"dimension": "conciseness", "preferred_arm": "control"},
        {"dimension": "conciseness", "preferred_arm": "control"},
    ])
    summary = pairwise_summary(df)
    q = summary[summary["dimension"] == "quality"].iloc[0]
    assert q["treatment"] == 2
    assert q["control"] == 1
    assert q["tied"] == 1
    c = summary[summary["dimension"] == "conciseness"].iloc[0]
    assert c["control"] == 2
    assert c["treatment_win_rate_decisive"] == 0.0
