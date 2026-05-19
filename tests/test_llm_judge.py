"""Tests for the LLM-judge parsing layer. Does not hit the live API."""

from __future__ import annotations

import pytest

from src.eval.llm_judge import JudgeResult, _parse_judge_response


def test_parses_plain_json():
    text = '{"quality": 4, "scope": 5, "honesty": 3, "tone": 4, "rationale": "good answer"}'
    r = _parse_judge_response(text)
    assert isinstance(r, JudgeResult)
    assert r.quality == 4
    assert r.scope == 5
    assert r.honesty == 3
    assert r.tone == 4
    assert "good answer" in r.rationale


def test_parses_json_inside_markdown_fence():
    text = '```json\n{"quality": 5, "scope": 5, "honesty": 5, "tone": 5, "rationale": "ideal"}\n```'
    r = _parse_judge_response(text)
    assert r.quality == 5
    assert r.rationale == "ideal"


def test_clips_out_of_range_scores():
    text = '{"quality": 9, "scope": -3, "honesty": 4, "tone": 1, "rationale": "x"}'
    r = _parse_judge_response(text)
    assert r.quality == 5
    assert r.scope == 0
    assert r.honesty == 4
    assert r.tone == 1


def test_handles_string_scores():
    text = '{"quality": "3", "scope": "4", "honesty": "5", "tone": "2", "rationale": "x"}'
    r = _parse_judge_response(text)
    assert r.quality == 3
    assert r.scope == 4


def test_extracts_json_amid_preamble():
    text = "Sure, here is the score:\n{\"quality\": 2, \"scope\": 3, \"honesty\": 3, \"tone\": 2, \"rationale\": \"ok\"}"
    r = _parse_judge_response(text)
    assert r.quality == 2
    assert r.honesty == 3


def test_raises_on_unparseable():
    with pytest.raises(Exception):
        _parse_judge_response("not json at all, no braces here")
