"""
Ticket generator for experiments.

Produces a deterministic stream of realistic e-commerce support tickets
that an agent processes during a long-running session. The same seed
produces the same sequence so treatment and control arms see identical
input streams.

The generator is intentionally simple: a bank of templates parameterized
by customer id and order id, with optional variants (tone, edge cases).
Real production support data would be noisier; this is enough to give
the agent meaningful work without confounding the experiment with
ticket-quality variance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class Ticket:
    ticket_template_id: str
    subject: str
    body: str
    customer_id: str
    related_order_id: str | None = None
    priority: str = "normal"
    tone_variant: str = "neutral"


_TONE_PREFIXES = {
    "neutral": "",
    "frustrated": "I am extremely frustrated. ",
    "polite": "Hi there, hope this finds you well. ",
    "terse": "",
    "confused": "I'm not sure what's going on but ",
}

_TONE_SUFFIXES = {
    "neutral": "",
    "frustrated": " Please help, this is unacceptable.",
    "polite": " Thank you so much for your time.",
    "terse": "",
    "confused": " Please let me know what's happening.",
}


# Each template is (id, subject_fmt, body_fmt, requires_order, default_priority).
_TEMPLATES: list[dict] = [
    {
        "id": "shipment_status_check",
        "subject": "Where is my order {order_id}?",
        "body": "Hi, I placed order {order_id} and I'd like to know its current status. Can you check?",
        "requires_order": True,
        "priority": "normal",
    },
    {
        "id": "damaged_item",
        "subject": "Damaged item in order {order_id}",
        "body": "My order {order_id} arrived but one of the items is damaged. I have photos. I would like a replacement.",
        "requires_order": True,
        "priority": "high",
    },
    {
        "id": "wrong_item_shipped",
        "subject": "Wrong item in order {order_id}",
        "body": "I received order {order_id} but it contains an item I did not order. How do we fix this?",
        "requires_order": True,
        "priority": "high",
    },
    {
        "id": "refund_status_check",
        "subject": "Refund status",
        "body": "I requested a refund a few days ago and haven't seen it in my account yet. Can you check on the status?",
        "requires_order": False,
        "priority": "normal",
    },
    {
        "id": "return_question",
        "subject": "How do I return order {order_id}?",
        "body": "I'd like to return order {order_id} since the product wasn't what I expected. What's the process?",
        "requires_order": True,
        "priority": "normal",
    },
    {
        "id": "shipping_policy_question",
        "subject": "Shipping options",
        "body": "What shipping options do you offer and how long do they take? Is there free shipping?",
        "requires_order": False,
        "priority": "low",
    },
    {
        "id": "account_password_reset",
        "subject": "Cannot reset password",
        "body": "I tried to reset my password but I'm not getting the reset email. My account is registered under this email.",
        "requires_order": False,
        "priority": "normal",
    },
    {
        "id": "billing_question",
        "subject": "Subscription billing question",
        "body": "I see a charge on my card I don't recognize. Can you tell me what it's for?",
        "requires_order": False,
        "priority": "normal",
    },
    {
        "id": "product_compatibility",
        "subject": "Is the {order_id} item compatible with my device?",
        "body": "Will the items in order {order_id} work with a 2023 MacBook Pro? I want to be sure before I open the box.",
        "requires_order": True,
        "priority": "low",
    },
    {
        "id": "cancel_order_request",
        "subject": "Cancel order {order_id}",
        "body": "I'd like to cancel order {order_id} if it hasn't shipped yet. Please confirm if you can do this.",
        "requires_order": True,
        "priority": "high",
    },
    {
        "id": "discount_request",
        "subject": "Discount request",
        "body": "I've been a customer for a while and was hoping for a discount on my next order. Is that something you can offer?",
        "requires_order": False,
        "priority": "low",
    },
    {
        "id": "tracking_no_movement",
        "subject": "Tracking has not updated",
        "body": "The tracking for order {order_id} hasn't moved in over a week. Is something wrong?",
        "requires_order": True,
        "priority": "normal",
    },
    {
        "id": "ambiguous_problem",
        "subject": "It's not working",
        "body": "It's not working. Can you fix it.",
        "requires_order": False,
        "priority": "normal",
    },
    {
        "id": "wrong_address",
        "subject": "Update shipping address for {order_id}",
        "body": "I gave the wrong shipping address for order {order_id}. Can you change it before it ships?",
        "requires_order": True,
        "priority": "high",
    },
    {
        "id": "product_defect_general",
        "subject": "Product stopped working",
        "body": "A product I bought a couple weeks ago has stopped working. I'd like a replacement or refund.",
        "requires_order": False,
        "priority": "normal",
    },
]


# Pool of customer / order pairs known to exist in the mock customer_db
# and order_system. The generator pulls realistic combinations from here.
_CUSTOMER_ORDER_POOL: list[tuple[str, str | None]] = [
    ("cust_001", "ord_a1b2"),
    ("cust_001", "ord_c3d4"),
    ("cust_001", None),
    ("cust_002", "ord_e5f6"),
    ("cust_002", None),
    ("cust_003", "ord_g7h8"),
    ("cust_003", None),
    ("cust_004", None),
    ("cust_005", "ord_i9j0"),
    ("cust_005", None),
]


@dataclass
class TicketGenerator:
    """
    Deterministic stream of tickets.

    Iterating over `generate(n)` yields `n` tickets sampled from the
    template bank, paired with realistic customer/order ids, varied in
    tone. Seeded for reproducibility.
    """

    seed: int = 42
    include_tone_variants: bool = True

    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def generate(self, n: int) -> Iterator[Ticket]:
        for _ in range(n):
            yield self._sample_one()

    def _sample_one(self) -> Ticket:
        template = self._rng.choice(_TEMPLATES)
        customer_id, order_id = self._sample_customer_order(
            requires_order=template["requires_order"]
        )

        tone = "neutral"
        if self.include_tone_variants and self._rng.random() < 0.4:
            tone = self._rng.choice(["frustrated", "polite", "terse", "confused"])

        subject = template["subject"].format(order_id=order_id or "")
        body_core = template["body"].format(order_id=order_id or "")
        body = (
            _TONE_PREFIXES[tone]
            + body_core
            + _TONE_SUFFIXES[tone]
        ).strip()

        return Ticket(
            ticket_template_id=template["id"],
            subject=subject,
            body=body,
            customer_id=customer_id,
            related_order_id=order_id,
            priority=template["priority"],
            tone_variant=tone,
        )

    def _sample_customer_order(self,
                               requires_order: bool) -> tuple[str, str | None]:
        if requires_order:
            candidates = [(c, o) for c, o in _CUSTOMER_ORDER_POOL if o is not None]
        else:
            candidates = list(_CUSTOMER_ORDER_POOL)
        return self._rng.choice(candidates)
