#!/bin/sh
# Prepare the mounted vault, then hand over to the command.
#
# The vault is a git repository the container writes to (R4.7) but does not own:
# it is bind-mounted from the host and must remain usable there after the
# container is gone (R7.6). So this only ever adds what git needs to operate --
# it never rewrites history, and never touches documents.
set -e

VAULT="${LLMDEX_VAULT:-/vault}"

if [ ! -d "$VAULT" ]; then
    echo "vault: $VAULT does not exist; creating it"
    mkdir -p "$VAULT"
fi

# Bind mounts arrive owned by the host user, which git refuses to act on.
git config --global --add safe.directory "$VAULT" 2>/dev/null || true

if [ ! -d "$VAULT/.git" ]; then
    echo "vault: initialising a git repository in $VAULT"
    git init -q "$VAULT"
fi

# Only set an identity if the repository has none, so a host-configured author
# is left alone.
if ! git -C "$VAULT" config user.email >/dev/null 2>&1; then
    git -C "$VAULT" config user.email "llmdex@localhost"
    git -C "$VAULT" config user.name "llmdex"
fi

if [ -z "${LLMDEX_LLM_BASE_URL:-}" ]; then
    echo "note: no LLMDEX_LLM_BASE_URL set - ingest and reads work, extraction returns 503"
fi

exec "$@"
