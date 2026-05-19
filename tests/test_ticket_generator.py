"""Tests for the ticket generator."""

from __future__ import annotations

from src.probes.ticket_generator import Ticket, TicketGenerator


def test_generator_produces_requested_count():
    gen = TicketGenerator(seed=42)
    tickets = list(gen.generate(15))
    assert len(tickets) == 15
    for t in tickets:
        assert isinstance(t, Ticket)
        assert t.customer_id
        assert t.subject
        assert t.body
        assert t.priority in {"low", "normal", "high", "urgent"}
        assert t.tone_variant in {"neutral", "frustrated", "polite", "terse", "confused"}


def test_generator_is_deterministic_under_seed():
    a = list(TicketGenerator(seed=1).generate(10))
    b = list(TicketGenerator(seed=1).generate(10))
    assert [t.ticket_template_id for t in a] == [t.ticket_template_id for t in b]
    assert [t.customer_id for t in a] == [t.customer_id for t in b]
    assert [t.body for t in a] == [t.body for t in b]


def test_generator_different_seeds_produce_different_streams():
    a = list(TicketGenerator(seed=1).generate(20))
    b = list(TicketGenerator(seed=2).generate(20))
    # Not strictly guaranteed to differ but overwhelmingly likely.
    assert (
        [t.ticket_template_id for t in a]
        != [t.ticket_template_id for t in b]
    )


def test_order_required_templates_get_order_ids():
    gen = TicketGenerator(seed=42)
    for ticket in gen.generate(40):
        if ticket.ticket_template_id in {
            "shipment_status_check",
            "damaged_item",
            "wrong_item_shipped",
            "return_question",
            "product_compatibility",
            "cancel_order_request",
            "tracking_no_movement",
            "wrong_address",
        }:
            assert ticket.related_order_id is not None, (
                f"Template {ticket.ticket_template_id} needs an order id"
            )


def test_tone_variants_appear_with_variants_enabled():
    gen = TicketGenerator(seed=7, include_tone_variants=True)
    tones = {t.tone_variant for t in gen.generate(60)}
    assert "neutral" in tones
    assert len(tones) > 1


def test_tone_variants_off():
    gen = TicketGenerator(seed=7, include_tone_variants=False)
    tones = {t.tone_variant for t in gen.generate(30)}
    assert tones == {"neutral"}
