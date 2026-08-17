"""Validate every document in the vault against the schema (R7.5).

The vault is hand-edited by design (R4.6), so the schema is only a contract if
something checks it. This is that check. It prints every failure rather than the
first, because a person fixing a vault wants the whole list.

    uv run python -m api.validate
"""

import sys

from api.store import store_from_env


def main() -> int:
    store = store_from_env()
    failures = store.validate_all()
    if not failures:
        return 0  # silence on success, so stderr carries only real failures
    for path, error in failures:
        print(f"{path}: {error}", file=sys.stderr)
    print(f"{len(failures)} invalid document(s) in {store.root}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
