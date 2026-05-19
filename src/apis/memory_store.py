"""
Mock memory store for the support agent.

This is the substrate of point 2 (directed learning from experience) and
point 1 (persistent identity). The agent writes structured notes here
after each ticket close; future ticket handling can retrieve relevant
notes.

For the pilot, this is a JSON-backed store with a small set of operations.
A real production system would use a vector database with semantic search.
We deliberately keep this layer simple so the experiments can attribute
behaviour to the agent rather than to retrieval quality.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryStore:
    """
    JSON-backed memory store.

    Each entry is a dict with at least:
      - id: a stable string
      - created_at: monotonic float
      - tags: list of strings (for filter retrieval)
      - body: the actual note content
      - context: optional dict capturing the situation in which the
        note was made (so we can apply point 9, generalization
        discipline: only re-use a note when context matches)
    """

    path: Path

    _entries: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            self._entries = json.loads(self.path.read_text())
        else:
            self._entries = []
        self._loaded = True

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2))

    # --- Public surface --------------------------------------------------

    def write(self,
              body: str,
              tags: list[str] | None = None,
              context: dict[str, Any] | None = None) -> str:
        """Append a new memory entry. Returns the entry id."""
        self._ensure_loaded()
        entry_id = f"mem_{int(time.time() * 1000)}_{len(self._entries)}"
        entry = {
            "id": entry_id,
            "created_at": time.time(),
            "tags": tags or [],
            "body": body,
            "context": context or {},
        }
        self._entries.append(entry)
        self._persist()
        return entry_id

    def search(self,
               tag: str | None = None,
               text: str | None = None,
               limit: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve memory entries.

        - tag: if given, only entries that include this tag.
        - text: case-insensitive substring match on the body.
        - limit: cap on returned entries (most recent first).
        """
        self._ensure_loaded()
        results = list(self._entries)
        if tag is not None:
            results = [e for e in results if tag in e.get("tags", [])]
        if text is not None:
            t = text.lower()
            results = [e for e in results if t in e.get("body", "").lower()]
        results.sort(key=lambda e: e["created_at"], reverse=True)
        return results[:limit]

    def list_all(self) -> list[dict[str, Any]]:
        """Return every entry, most recent first."""
        self._ensure_loaded()
        return sorted(self._entries, key=lambda e: e["created_at"], reverse=True)

    def count(self) -> int:
        """How many entries are stored."""
        self._ensure_loaded()
        return len(self._entries)

    def clear(self) -> None:
        """Erase all entries. Used in tests and experiment setup."""
        self._entries = []
        self._loaded = True
        self._persist()
