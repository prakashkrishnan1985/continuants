"""
Token-cost tracking for the experiment runs.

Provides a `CostTracker` that records cumulative token usage and
estimates dollar cost, plus a simple price table for the models we use.

Pricing source: Anthropic's published API pricing. Values can be
overridden via environment variables for sensitivity analyses.

The tracker is *advisory*. It estimates spend so the runner can abort
before exceeding a budget, but the canonical billing source is
Anthropic's dashboard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# Per-million-token prices in USD. Conservative defaults; override via
# env if Anthropic adjusts pricing.
_DEFAULT_PRICE_TABLE: dict[str, dict[str, float]] = {
    # Claude Sonnet 4.6 family
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    # Claude Opus 4.7 family
    "claude-opus-4-7": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    # Claude Haiku 4.5 family
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
}


def _override_from_env(model: str, kind: str) -> float | None:
    var = f"PRICE_{model.upper().replace('-', '_')}_{kind.upper()}"
    raw = os.environ.get(var)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def price_per_million(model: str, kind: str) -> float:
    """Return USD per million tokens for (model, kind). 0.0 if unknown."""
    override = _override_from_env(model, kind)
    if override is not None:
        return override
    table = _DEFAULT_PRICE_TABLE.get(_normalize_model(model), {})
    return table.get(kind, 0.0)


def _normalize_model(model: str) -> str:
    """
    Map specific version strings to family keys in the price table.

    The pricing model is stable across minor versions, so we collapse
    e.g. claude-sonnet-4-6-20260301 to claude-sonnet-4-6.
    """
    if not model:
        return model
    for family in _DEFAULT_PRICE_TABLE.keys():
        if model.startswith(family):
            return family
    return model


@dataclass
class CostTracker:
    """Cumulative token / cost tracker for a single experiment session."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    api_calls: int = 0

    def record_usage(self, usage: Any) -> None:
        """
        Accept either an Anthropic SDK Usage object or a plain dict.
        Unknown fields are ignored.
        """
        if usage is None:
            return
        getter = (
            usage.get if isinstance(usage, dict)
            else lambda k, d=0: getattr(usage, k, d)
        )
        self.input_tokens += int(getter("input_tokens", 0) or 0)
        self.output_tokens += int(getter("output_tokens", 0) or 0)
        self.cache_write_tokens += int(getter("cache_creation_input_tokens", 0) or 0)
        self.cache_read_tokens += int(getter("cache_read_input_tokens", 0) or 0)
        self.api_calls += 1

    def estimate_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * price_per_million(self.model, "input")
            + self.output_tokens / 1_000_000 * price_per_million(self.model, "output")
            + self.cache_write_tokens / 1_000_000 * price_per_million(self.model, "cache_write")
            + self.cache_read_tokens / 1_000_000 * price_per_million(self.model, "cache_read")
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "estimated_usd": round(self.estimate_usd(), 4),
        }
