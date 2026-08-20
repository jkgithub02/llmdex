"""The Hugging Face fetch layer (R1.2, R1.6).

Everything that touches the network lives here, so every other module can be
tested offline against a :class:`RepoSnapshot` built from a fixture.

Failures are named rather than lumped together: "gated", "private" and "does not
exist" call for three different actions from whoever ran the ingest, and telling
them only that it "failed" wastes their time (R1.6).
"""

import json
import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.common.exceptions import (
    AccessUndetermined,
    GatedRepo,
    IngestError,
    PrivateRepo,
    RepoNotFound,
)

# Re-exported so `from app.features.models.fetch import RepoNotFound` keeps
# working: these are raised here, they simply are not defined here.
__all__ = [
    "AccessUndetermined",
    "GatedRepo",
    "IngestError",
    "PrivateRepo",
    "RepoNotFound",
    "RepoSnapshot",
    "fetch_revision",
    "fetch_snapshot",
    "fetch_tensor_headers",
]

from app.features.models.derive import decoder_config, packed_quantization_bits
from app.features.models.tensors import header_length, parse_header

HF_BASE = "https://huggingface.co"
TIMEOUT = 60.0

#: How much of a shard to ask for when reading its header. The header is a u64
#: length followed by that much JSON; one megabyte covers the largest seen (a
#: 128-expert layer runs to a few hundred kilobytes) and a longer one is fetched
#: exactly rather than guessed around.
HEADER_WINDOW = 1 << 20


class RepoSnapshot(BaseModel):
    """Everything ingest needs, fetched once, so derivation stays pure."""

    model_id: str
    revision: str | None = None
    siblings: list[dict[str, Any]] = Field(default_factory=list)
    config: dict[str, Any] | None = None
    generation_config: dict[str, Any] | None = None
    readme: str | None = None
    safetensors_total: int | None = None
    #: One parsed safetensors header per shard, read only when the summed total
    #: above cannot be a parameter count (R2.2). ``None`` means not read.
    tensor_headers: list[dict[str, Any]] | None = None


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _headers() -> dict[str, str]:
    token = _token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _is_gated_message(body: str) -> bool:
    lowered = body.lower()
    return "gated" in lowered or "is restricted" in lowered or "accept" in lowered


def _raise_for_repo(model_id: str, status: int, body: str) -> None:
    """R1.6 - name the failure, including when the honest name is "cannot tell"."""
    if status == 404:
        raise RepoNotFound(
            f"{model_id} does not exist on Hugging Face. "
            "Check the spelling, including the vendor prefix."
        )

    if status in (401, 403):
        if _is_gated_message(body):
            raise GatedRepo(
                f"{model_id} is gated. Accept the terms at {HF_BASE}/{model_id} "
                "and set HF_TOKEN to a token belonging to the account that accepted them."
            )
        if _token() is None:
            # Anonymous callers get the same 401 for missing and for private, so
            # claiming either one would be a guess dressed as a diagnosis.
            raise AccessUndetermined(
                f"Hugging Face refused access to {model_id} without saying why. "
                "It either does not exist, or it is private or gated -- an anonymous "
                "request cannot tell these apart. Set HF_TOKEN and retry to get a "
                "definite answer."
            )
        raise PrivateRepo(
            f"{model_id} is private, or your HF_TOKEN does not grant read access to it."
        )

    raise IngestError(f"Hugging Face returned HTTP {status} for {model_id}: {body[:200]}")


def fetch_snapshot(model_id: str, client: httpx.Client | None = None) -> RepoSnapshot:
    """Fetch model info, config, and the file listing with real byte sizes (R1.2)."""
    owns_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
    try:
        info = client.get(
            f"{HF_BASE}/api/models/{model_id}", params={"blobs": "true"}, headers=_headers()
        )
        if info.status_code != 200:
            _raise_for_repo(model_id, info.status_code, info.text)
        payload = info.json()

        snapshot = RepoSnapshot(
            model_id=payload.get("id", model_id),
            revision=payload.get("sha"),
            siblings=payload.get("siblings", []),
            safetensors_total=(payload.get("safetensors") or {}).get("total"),
        )

        snapshot.config = _optional_json(client, model_id, "config.json")
        snapshot.generation_config = _optional_json(client, model_id, "generation_config.json")
        snapshot.readme = _optional_text(client, model_id, "README.md")

        # Only for a checkpoint whose reported total counts packed containers
        # rather than parameters. Everywhere else the Hub's number is right and
        # this would be two range requests per shard spent to confirm it.
        #
        # Through decoder_config, because a multimodal wrapper nests its
        # quantization_config under text_config. Reading the raw config here
        # while derive reads the unwrapped one is how Kimi K3 ended up with no
        # parameter count at all: this skipped the headers, and derive then
        # found a 4-bit checkpoint with nothing to unpack it with.
        if packed_quantization_bits(decoder_config(snapshot.config or {})) is not None:
            snapshot.tensor_headers = fetch_tensor_headers(client, model_id)
        return snapshot
    finally:
        if owns_client:
            client.close()


def _optional_text(client: httpx.Client, model_id: str, filename: str) -> str | None:
    """Fetch a file that may legitimately be absent.

    Only **404** means absent. A gated repository serves its metadata publicly but
    answers 401 on the files themselves, and treating that denial as absence would
    produce a document full of nulls that never records the real reason -- looking
    exactly like an honest GGUF-only repo while being nothing of the kind.
    Anything that is not a 200 or a 404 is raised (R1.6).
    """
    # resolve/, not raw/: `raw` serves the git blob, and for an LFS-tracked file
    # that is a 133-byte pointer rather than the content. Kimi K3's 57 MB
    # safetensors index is LFS-tracked, so this returned
    # "version https://git-lfs.github.com/spec/v1 ..." and the caller reported
    # a valid file as unparseable. `resolve` serves content for both kinds, and
    # is already what _range uses for exactly this reason.
    r = client.get(f"{HF_BASE}/{model_id}/resolve/main/{filename}", headers=_headers())
    if r.status_code == 200:
        return r.text
    if r.status_code == 404:
        return None
    _raise_for_repo(model_id, r.status_code, r.text)
    return None  # unreachable; _raise_for_repo always raises


def _optional_json(client: httpx.Client, model_id: str, filename: str) -> dict[str, Any] | None:
    text = _optional_text(client, model_id, filename)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        # Present but unparseable is a different fact from absent, and silently
        # returning None would erase the difference.
        raise IngestError(f"{model_id}/{filename} is present but is not valid JSON") from exc


def fetch_tensor_headers(client: httpx.Client, model_id: str) -> list[dict[str, Any]] | None:
    """Every shard's safetensors header, or ``None`` if the repo has no safetensors.

    Reads the header only -- the tensor data behind it is hundreds of gigabytes
    and is never touched. A shard that answers anything other than 200/206 is
    raised rather than skipped: a missing header does not make the count
    smaller, it makes it wrong (R1.5).
    """
    index = _optional_json(client, model_id, "model.safetensors.index.json")
    if index is not None:
        shards = sorted(set((index.get("weight_map") or {}).values()))
    else:
        shards = ["model.safetensors"]

    headers = []
    for shard in shards:
        blob = _range(client, model_id, shard, 0, HEADER_WINDOW - 1)
        if blob is None:
            # Only reachable for the single-file guess above; an indexed shard
            # that is absent means the index is lying, and _range raises.
            return None
        try:
            headers.append(parse_header(blob))
        except ValueError:
            # The header is longer than the window. Read its declared length and
            # fetch exactly that, rather than growing the window by guesswork.
            length = header_length(blob)
            exact = _range(client, model_id, shard, 8, 8 + length - 1)
            headers.append(json.loads(exact or b"{}"))
    return headers or None


def _range(
    client: httpx.Client, model_id: str, filename: str, start: int, end: int
) -> bytes | None:
    """A byte range of a repository file. 404 is absent; anything else raises."""
    r = client.get(
        f"{HF_BASE}/{model_id}/resolve/main/{filename}",
        headers={**_headers(), "Range": f"bytes={start}-{end}"},
    )
    if r.status_code in (200, 206):
        return r.content
    if r.status_code == 404:
        return None
    _raise_for_repo(model_id, r.status_code, r.text)
    return None  # unreachable; _raise_for_repo always raises


def fetch_revision(model_id: str, client: httpx.Client | None = None) -> str | None:
    """Just the current commit SHA, for drift detection (R6.6)."""
    owns_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
    try:
        r = client.get(f"{HF_BASE}/api/models/{model_id}", headers=_headers())
        if r.status_code != 200:
            _raise_for_repo(model_id, r.status_code, r.text)
        return r.json().get("sha")
    finally:
        if owns_client:
            client.close()
