"""
Real-data ticket generator using the Bitext Customer Support dataset.

The Bitext Customer Support LLM Chatbot Training Dataset (Hugging Face:
`bitext/Bitext-customer-support-llm-chatbot-training-dataset`) is a
public dataset of ~27,000 customer support exchanges across 11 intent
categories. Using it instead of our hand-crafted templates addresses
the input-realism limitation noted in the paper: real customer language
is messier, more varied, and more likely to surface drift signals that
template-based synthetic data might suppress.

This generator yields the same `Ticket` shape the experiment runner
consumes, so it is a drop-in replacement for `TicketGenerator`.

Setup (one-time):

    pip install datasets
    python -c "from datasets import load_dataset; \\
               ds = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset')['train']; \\
               ds.to_csv('data/bitext.csv')"

Then point the runner at `data/bitext.csv`.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.probes.ticket_generator import Ticket


# Bitext exchanges contain templated placeholders like {{Order Number}}.
# We substitute the ones we have realistic values for so the agent has
# something to actually look up. Unknown placeholders are left intact;
# the agent's response to them is itself a useful behavioural signal.
def _substitute_placeholders(text: str,
                              order_id: str | None,
                              customer_id: str) -> str:
    if not text:
        return text
    substitutions = {
        r"\{\{Order Number\}\}": order_id or "[no order on file]",
        r"\{\{Order ID\}\}": order_id or "[no order on file]",
        r"\{\{Order_Number\}\}": order_id or "[no order on file]",
        r"\{\{Customer ID\}\}": customer_id,
        r"\{\{Customer_ID\}\}": customer_id,
        r"\{\{Account Number\}\}": customer_id,
    }
    out = text
    for pattern, replacement in substitutions.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


# Map Bitext intent categories to our priority levels.
_CATEGORY_PRIORITY = {
    "ACCOUNT": "normal",
    "CANCELLATION_FEE": "high",
    "CONTACT": "low",
    "DELIVERY": "normal",
    "FEEDBACK": "low",
    "INVOICE": "normal",
    "NEWSLETTER": "low",
    "ORDER": "normal",
    "PAYMENT": "high",
    "REFUND": "high",
    "SHIPPING_ADDRESS": "high",
}

# Bitext exposes tone flags via a `flags` column with letter codes.
# Map a subset to our existing tone variants.
_FLAG_TO_TONE = {
    "P": "polite",
    "Q": "neutral",
    "L": "neutral",
    "C": "neutral",
}


# Pool of customer / order pairs known to exist in our mock APIs.
# Bitext does not include identifiers, so we attach them deterministically.
_CUSTOMER_POOL = ["cust_001", "cust_002", "cust_003", "cust_004", "cust_005"]
_ORDER_POOL_BY_CUSTOMER: dict[str, list[str]] = {
    "cust_001": ["ord_a1b2", "ord_c3d4"],
    "cust_002": ["ord_e5f6"],
    "cust_003": ["ord_g7h8"],
    "cust_004": [],
    "cust_005": ["ord_i9j0"],
}


def _derive_priority(category: str) -> str:
    return _CATEGORY_PRIORITY.get((category or "").upper(), "normal")


def _derive_tone(flags: str) -> str:
    if not flags:
        return "neutral"
    for ch in flags:
        if ch in _FLAG_TO_TONE:
            return _FLAG_TO_TONE[ch]
    return "neutral"


def _derive_subject(instruction: str, category: str | None) -> str:
    """Pick a short subject from the instruction's first clause."""
    text = (instruction or "").strip().replace("\n", " ")
    # Use the first sentence, capped at 80 chars.
    for sep in (". ", "? ", "! "):
        if sep in text:
            head, _ = text.split(sep, 1)
            text = head
            break
    text = text[:80].strip()
    if category and len(text) < 8:
        return f"{category.title()} request"
    return text or "Support request"


@dataclass
class BitextTicketGenerator:
    """
    Deterministic stream of tickets sampled from a Bitext CSV.

    Parameters
    ----------
    csv_path
        Path to the local CSV exported from the Bitext dataset.
        Expected columns: `instruction`, `category`, optionally `intent`
        and `flags`.
    seed
        Random seed. Same seed + same csv yields identical samples.
    """

    csv_path: Path
    seed: int = 42

    _rng: random.Random = field(init=False, repr=False)
    _rows: list[dict[str, str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._rows = self._load_rows()
        if not self._rows:
            raise ValueError(
                f"No usable rows found in {self.csv_path}. "
                "Expected columns include `instruction` and `category`."
            )

    def _load_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("instruction"):
                    continue
                rows.append(row)
        return rows

    # --- Public surface --------------------------------------------------

    def generate(self, n: int) -> Iterator[Ticket]:
        # Sample without replacement so we don't repeat the same exchange
        # within one session.
        indices = self._rng.sample(range(len(self._rows)), k=min(n, len(self._rows)))
        for idx in indices:
            yield self._build_ticket(self._rows[idx])

    def _build_ticket(self, row: dict[str, str]) -> Ticket:
        instruction_raw = row.get("instruction", "").strip()
        category = row.get("category", "").strip()
        intent = row.get("intent", "").strip() or category or "general"
        flags = row.get("flags", "").strip()

        customer_id = self._rng.choice(_CUSTOMER_POOL)
        order_pool = _ORDER_POOL_BY_CUSTOMER.get(customer_id, [])
        related_order_id = self._rng.choice(order_pool) if order_pool else None

        instruction = _substitute_placeholders(
            instruction_raw,
            order_id=related_order_id,
            customer_id=customer_id,
        )

        return Ticket(
            ticket_template_id=f"bitext::{intent.lower().replace(' ', '_')}",
            subject=_derive_subject(instruction, category),
            body=instruction,
            customer_id=customer_id,
            related_order_id=related_order_id,
            priority=_derive_priority(category),
            tone_variant=_derive_tone(flags),
        )
