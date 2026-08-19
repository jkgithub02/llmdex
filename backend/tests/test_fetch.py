"""The fetch layer's failure modes (R1.6).

Driven by what Hugging Face actually returns, captured live on 2026-08-17:

- A repository that does not exist answers **401** ``{"error":"Invalid username
  or password."}`` when unauthenticated. It deliberately will not tell an
  anonymous caller whether a repo is missing or merely invisible.
- A *gated* repository answers **200** for its metadata but **401** for
  ``config.json`` with "Access to model X is restricted".

The second case is the dangerous one. Treating that 401 as "no config.json here"
produces a document whose fields are all null for a reason it never records --
indistinguishable from an honest GGUF-only repo.
"""

import json
import struct

import httpx
import pytest

from backend.models.fetch import (
    AccessUndetermined,
    GatedRepo,
    RepoNotFound,
    fetch_snapshot,
)

HF = "https://huggingface.co"


def transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_nonexistent_repo_unauthenticated_does_not_claim_to_know_why(monkeypatch):
    """R1.6 - honest about the limit: anonymous callers cannot tell missing from private."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    def handler(request):
        return httpx.Response(401, json={"error": "Invalid username or password."})

    with pytest.raises(AccessUndetermined) as exc, transport(handler) as client:
        fetch_snapshot("nobody/nothing", client=client)

    message = str(exc.value)
    assert "HF_TOKEN" in message, "must say how to disambiguate"
    assert "does not exist" in message and "private" in message, "must name the possibilities"


def test_explicit_404_is_a_clean_not_found(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")

    def handler(request):
        return httpx.Response(404, json={"error": "Repo not found"})

    with pytest.raises(RepoNotFound), transport(handler) as client:
        fetch_snapshot("nobody/nothing", client=client)


def test_gated_config_is_an_error_not_a_missing_file(monkeypatch):
    """The bug live verification caught: 401 on config.json is denial, not absence."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    def handler(request):
        if "/api/models/" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "id": "meta-llama/Llama-3.1-8B",
                    "sha": "abc123",
                    "siblings": [{"rfilename": "model.safetensors", "size": 100}],
                },
            )
        return httpx.Response(
            401,
            text="Access to model meta-llama/Llama-3.1-8B is restricted. "
            "You must have access to it and be authenticated to access it.",
        )

    with pytest.raises(GatedRepo) as exc, transport(handler) as client:
        fetch_snapshot("meta-llama/Llama-3.1-8B", client=client)
    assert "gated" in str(exc.value).lower() or "restricted" in str(exc.value).lower()


def test_genuinely_absent_config_is_still_none(monkeypatch):
    """A GGUF-only repo really has no config.json. 404 means absent, and that is fine."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def handler(request):
        if "/api/models/" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "id": "Qwen/Qwen3-8B-GGUF",
                    "sha": "def456",
                    "siblings": [{"rfilename": "m-Q4_K_M.gguf", "size": 100}],
                },
            )
        return httpx.Response(404, text="Entry not found")

    with transport(handler) as client:
        snap = fetch_snapshot("Qwen/Qwen3-8B-GGUF", client=client)
    assert snap.config is None
    assert snap.revision == "def456"
    assert len(snap.siblings) == 1


def test_successful_fetch_captures_revision_and_sizes(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def handler(request):
        if "/api/models/" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "id": "a/b",
                    "sha": "cafe1234",
                    "siblings": [{"rfilename": "model.safetensors", "size": 4096}],
                    "safetensors": {"total": 2048},
                },
            )
        if request.url.path.endswith("config.json"):
            return httpx.Response(200, text='{"model_type": "llama"}')
        return httpx.Response(404)

    with transport(handler) as client:
        snap = fetch_snapshot("a/b", client=client)
    assert snap.revision == "cafe1234"
    assert snap.safetensors_total == 2048
    assert snap.config == {"model_type": "llama"}


def test_server_error_is_not_mistaken_for_a_missing_file(monkeypatch):
    """A 503 on config.json must not silently become 'this repo has no config'."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def handler(request):
        if "/api/models/" in str(request.url):
            return httpx.Response(200, json={"id": "a/b", "sha": "x", "siblings": []})
        return httpx.Response(503, text="upstream unavailable")

    with pytest.raises(Exception) as exc, transport(handler) as client:
        fetch_snapshot("a/b", client=client)
    assert "503" in str(exc.value)


# --------------------------------------------------------------------------
# R2.2 - the safetensors headers, read only when the summed total is unusable
# --------------------------------------------------------------------------

PACKED_CONFIG = {
    "model_type": "nemotron_h",
    "quantization_config": {"config_groups": {"g": {"weights": {"num_bits": 4}}}},
}


def _shard(tensors: dict) -> bytes:
    body = json.dumps(tensors).encode()
    return struct.pack("<Q", len(body)) + body


def _hub(request, *, config: dict, files: dict[str, bytes]):
    """A Hub that serves one repo, honouring Range on the weight files."""
    if "/api/models/" in str(request.url):
        return httpx.Response(
            200,
            json={"id": "a/b", "sha": "cafe", "siblings": [], "safetensors": {"total": 999}},
        )
    name = request.url.path.rsplit("/", 1)[-1]
    if name == "config.json":
        return httpx.Response(200, text=json.dumps(config))
    if name in files:
        blob = files[name]
        header = request.headers.get("Range")
        if header:
            start, end = header.removeprefix("bytes=").split("-")
            return httpx.Response(206, content=blob[int(start) : int(end) + 1])
        return httpx.Response(200, content=blob)
    return httpx.Response(404, text="Entry not found")


def test_a_packed_checkpoint_has_its_headers_read(monkeypatch):
    """R2.2 - the one case where the Hub's total cannot be used is the one case
    worth spending range requests on."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    shard = _shard({"model.layers.0.mlp.up_proj.weight": {"dtype": "U8", "shape": [4, 2]}})

    def handler(request):
        return _hub(
            request,
            config=PACKED_CONFIG,
            files={
                "model.safetensors.index.json": json.dumps(
                    {"weight_map": {"model.layers.0.mlp.up_proj.weight": "model-00001.safetensors"}}
                ).encode(),
                "model-00001.safetensors": shard,
            },
        )

    with transport(handler) as client:
        snap = fetch_snapshot("a/b", client=client)

    assert snap.tensor_headers is not None
    assert snap.tensor_headers[0]["model.layers.0.mlp.up_proj.weight"]["shape"] == [4, 2]


def test_an_unquantized_checkpoint_spends_no_range_requests(monkeypatch):
    """The Hub's total is right for these, and a 60-shard model would cost 120
    requests to learn nothing new."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return _hub(request, config={"model_type": "llama"}, files={})

    with transport(handler) as client:
        snap = fetch_snapshot("a/b", client=client)

    assert snap.tensor_headers is None
    assert not any("safetensors" in path for path in seen)


def test_a_single_file_checkpoint_needs_no_index(monkeypatch):
    """Small models ship one `model.safetensors` and no index at all."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    shard = _shard({"model.embed_tokens.weight": {"dtype": "U8", "shape": [8, 4]}})

    def handler(request):
        return _hub(request, config=PACKED_CONFIG, files={"model.safetensors": shard})

    with transport(handler) as client:
        snap = fetch_snapshot("a/b", client=client)

    assert snap.tensor_headers is not None
    assert len(snap.tensor_headers) == 1


def test_no_safetensors_at_all_is_absent_not_an_error(monkeypatch):
    """A GGUF-only repo that also declares quantization has nothing to read."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def handler(request):
        return _hub(request, config=PACKED_CONFIG, files={})

    with transport(handler) as client:
        snap = fetch_snapshot("a/b", client=client)

    assert snap.tensor_headers is None


def test_a_refused_shard_is_raised_not_skipped(monkeypatch):
    """R1.5 - a header we were denied would silently shorten the count. Ingest
    fails naming the cause instead."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def handler(request):
        if request.url.path.endswith(".safetensors"):
            return httpx.Response(503, text="upstream unavailable")
        return _hub(
            request,
            config=PACKED_CONFIG,
            files={
                "model.safetensors.index.json": json.dumps(
                    {"weight_map": {"a.weight": "model-00001.safetensors"}}
                ).encode()
            },
        )

    with pytest.raises(Exception) as exc, transport(handler) as client:
        fetch_snapshot("a/b", client=client)
    assert "503" in str(exc.value)
