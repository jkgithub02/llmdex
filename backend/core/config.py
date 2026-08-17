"""Where the service reads its configuration from.

Everything here is resolved from the environment so that nothing about a
deployment is baked into the code (R7.4 says the same about the extraction
endpoint). Keeping it in one module means the CLI, the API and the tests all
agree on where the vault is without importing each other.
"""

import os
from pathlib import Path

from backend.core.store import Store

# backend/core/config.py -> backend/ -> repo root. The vault is a sibling of the
# package, not a child of it (R7.6: it is a separate git repository).
DEFAULT_VAULT = Path(__file__).resolve().parents[2] / "vault"


def store_from_env() -> Store:
    """The vault named by ``LLMDEX_VAULT``, or the sibling ``./vault`` directory."""
    return Store(os.environ.get("LLMDEX_VAULT", DEFAULT_VAULT))
