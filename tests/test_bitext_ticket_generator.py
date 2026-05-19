"""Tests for the Bitext ticket generator and the ticket-source factory."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.probes.bitext_ticket_generator import BitextTicketGenerator
from src.probes.ticket_source import build_ticket_source


def _write_fixture_csv(path: Path) -> None:
    rows = [
        {
            "instruction": "I want to know the status of my order. It has been over a week.",
            "category": "ORDER",
            "intent": "track_order",
            "flags": "Q",
        },
        {
            "instruction": "Please refund the charge on my card from last month. It was wrong.",
            "category": "REFUND",
            "intent": "request_refund",
            "flags": "L",
        },
        {
            "instruction": "Could you please help me change my delivery address?",
            "category": "SHIPPING_ADDRESS",
            "intent": "change_delivery_address",
            "flags": "P",
        },
        {
            "instruction": "Cancel my subscription right now.",
            "category": "ACCOUNT",
            "intent": "cancel_account",
            "flags": "C",
        },
        {
            "instruction": "My invoice is missing line items.",
            "category": "INVOICE",
            "intent": "request_invoice_correction",
            "flags": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "category", "intent", "flags"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_bitext_generator_loads_and_yields_tickets(tmp_path):
    csv_path = tmp_path / "bitext.csv"
    _write_fixture_csv(csv_path)

    gen = BitextTicketGenerator(csv_path=csv_path, seed=1)
    tickets = list(gen.generate(3))

    assert len(tickets) == 3
    for t in tickets:
        assert t.body
        assert t.subject
        assert t.customer_id.startswith("cust_")
        assert t.priority in {"low", "normal", "high"}
        assert t.tone_variant in {"neutral", "polite", "frustrated", "terse", "confused"}
        assert t.ticket_template_id.startswith("bitext::")


def test_bitext_generator_is_deterministic(tmp_path):
    csv_path = tmp_path / "bitext.csv"
    _write_fixture_csv(csv_path)

    a = list(BitextTicketGenerator(csv_path=csv_path, seed=42).generate(3))
    b = list(BitextTicketGenerator(csv_path=csv_path, seed=42).generate(3))
    assert [t.body for t in a] == [t.body for t in b]
    assert [t.customer_id for t in a] == [t.customer_id for t in b]


def test_bitext_generator_samples_without_replacement(tmp_path):
    csv_path = tmp_path / "bitext.csv"
    _write_fixture_csv(csv_path)

    gen = BitextTicketGenerator(csv_path=csv_path, seed=1)
    tickets = list(gen.generate(5))  # exactly the dataset size
    bodies = [t.body for t in tickets]
    assert len(set(bodies)) == len(bodies)  # all unique


def test_bitext_generator_empty_csv_raises(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("instruction,category,intent,flags\n")
    with pytest.raises(ValueError, match="No usable rows"):
        BitextTicketGenerator(csv_path=csv_path, seed=1)


def test_factory_default_returns_templated(tmp_path):
    stream = build_ticket_source("templated", seed=1)
    tickets = list(stream.generate(3))
    assert len(tickets) == 3
    # Templated generator produces template-ids that don't start with bitext::
    assert all(not t.ticket_template_id.startswith("bitext::") for t in tickets)


def test_factory_bitext_returns_bitext(tmp_path):
    csv_path = tmp_path / "bitext.csv"
    _write_fixture_csv(csv_path)
    stream = build_ticket_source("bitext", seed=1, bitext_csv=csv_path)
    tickets = list(stream.generate(2))
    assert all(t.ticket_template_id.startswith("bitext::") for t in tickets)


def test_factory_missing_bitext_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Bitext CSV not found"):
        build_ticket_source("bitext", seed=1, bitext_csv=tmp_path / "nope.csv")


def test_factory_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown ticket source"):
        build_ticket_source("magical", seed=1)
