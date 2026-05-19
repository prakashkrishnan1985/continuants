"""
Factory for picking a ticket-stream source.

Two sources are supported:

  - "templated" (default) — the original `TicketGenerator`. Synthetic
    template-based tickets, fully self-contained, no external data.

  - "bitext" — the `BitextTicketGenerator`. Loads real customer-support
    exchanges from a local CSV export of the Bitext dataset. More
    realistic input distribution, requires the dataset on disk.

The runner accepts a `--ticket-source` flag; this module dispatches to
the right generator. New sources can be added without touching the
runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

from src.probes.bitext_ticket_generator import BitextTicketGenerator
from src.probes.ticket_generator import Ticket, TicketGenerator


class TicketStream(Protocol):
    """Common interface every ticket source must satisfy."""

    def generate(self, n: int) -> Iterator[Ticket]: ...


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BITEXT_CSV = PROJECT_ROOT / "data" / "bitext.csv"


def build_ticket_source(name: str,
                        seed: int,
                        bitext_csv: Path | None = None) -> TicketStream:
    """
    Construct a ticket source by name.

    Raises ValueError on unknown name, or FileNotFoundError if bitext
    is selected but the CSV is missing.
    """
    name = (name or "templated").lower()
    if name in {"templated", "synthetic", "default"}:
        return TicketGenerator(seed=seed)
    if name in {"bitext", "real", "real-world"}:
        path = bitext_csv or DEFAULT_BITEXT_CSV
        if not path.exists():
            raise FileNotFoundError(
                f"Bitext CSV not found at {path}. Download with:\n"
                "  pip install datasets\n"
                "  python -c \"from datasets import load_dataset; "
                "ds = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset')['train']; "
                f"ds.to_csv('{path}')\""
            )
        return BitextTicketGenerator(csv_path=path, seed=seed)
    raise ValueError(
        f"Unknown ticket source: {name!r}. "
        "Valid: templated, bitext."
    )
