"""
Centralized environment loading.

Every experiment script calls `load_environment()` at startup. It reads
`.env` from the project root (which is gitignored) into the process's
environment so the Anthropic SDK can pick up `ANTHROPIC_API_KEY`
automatically.

Never read the API key directly here; never log it; never write it to
disk anywhere except `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_environment() -> None:
    """Load .env into os.environ. No-op if .env is absent."""
    if DOTENV_PATH.exists():
        load_dotenv(DOTENV_PATH, override=False)


def require(env_var: str) -> str:
    """Read an env var, raising a clear error if it is missing."""
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"Required environment variable {env_var} is not set. "
            f"Add it to {DOTENV_PATH} (which is gitignored)."
        )
    return value
