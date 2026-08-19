"""The HTTP surface (R6.3, R7.2, R7.3, R1.6).

Network is injected, so these run offline. The live end-to-end check lives in
``test_live.py`` and is the only thing here that touches Hugging Face.
"""

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.schemas import Checkpoint, ModelDoc
from app.core.store import Store
from app.main import app, get_fetcher, get_store
from app.models.fetch import GatedRepo, RepoNotFound, RepoSnapshot

FIXTURES = Path(__file__).parent / "fixtures"


def snapshot(name: str) -> RepoSnapshot:
    raw = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    info = raw["info"]
    return RepoSnapshot(
        model_id=raw["model_id"],
        revision=info.get("sha"),
        siblings=info.get("siblings", []),
        config=raw.get("config"),
        readme=raw.get("readme"),
        safetensors_total=(info.get("safetensors") or {}).get("total"),
    )


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    return Store(root)


@pytest.fixture
def client(vault):
    fixtures = {
        "Qwen/Qwen3-8B": "qwen3-8b",
        "nvidia/Nemotron-H-8B-Base-8K": "nemotron-h-8b",
        "Qwen/Qwen3-8B-GGUF": "qwen3-8b-gguf",
        "HuggingFaceTB/SmolLM2-135M": "smollm2-135m",
    }

    def fake_fetch(model_id: str) -> RepoSnapshot:
        if model_id == "nobody/nothing":
            raise RepoNotFound("nobody/nothing does not exist on Hugging Face")
        if model_id == "meta-llama/Gated-Thing":
            raise GatedRepo("meta-llama/Gated-Thing is gated. Accept the terms")
        if model_id not in fixtures:
            raise RepoNotFound(f"{model_id} not in fixtures")
        return snapshot(fixtures[model_id])

    app.dependency_overrides[get_store] = lambda: vault
    app.dependency_overrides[get_fetcher] = lambda: fake_fetch
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# health and schema
# ---------------------------------------------------------------------------


def test_health(client):
    assert client.get("/health").status_code == 200


def test_openapi_schema_is_published(client):
    """R7.3 - the frontend client is generated from this, never hand-written."""
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert "/models" in paths
    assert "/ingest" in paths


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def test_ingest_creates_a_model(client):
    r = client.post("/ingest", json={"model_id": "Qwen/Qwen3-8B"})
    assert r.status_code == 201
    body = r.json()
    assert body["model_id"] == "Qwen/Qwen3-8B"
    assert body["checkpoints"][0]["derived"]["architecture"] == "qwen3"


def test_ingested_model_is_then_readable(client):
    client.post("/ingest", json={"model_id": "Qwen/Qwen3-8B"})
    r = client.get("/models/Qwen/Qwen3-8B")
    assert r.status_code == 200
    assert r.json()["model_id"] == "Qwen/Qwen3-8B"


def test_ingest_of_missing_repo_is_404_and_says_so(client):
    """R1.6 - name which failure it was."""
    r = client.post("/ingest", json={"model_id": "nobody/nothing"})
    assert r.status_code == 404
    assert "does not exist" in r.json()["detail"]


def test_ingest_of_gated_repo_is_403_and_says_gated(client):
    r = client.post("/ingest", json={"model_id": "meta-llama/Gated-Thing"})
    assert r.status_code == 403
    assert "gated" in r.json()["detail"].lower()


def test_ingest_is_idempotent_over_http(client):
    first = client.post("/ingest", json={"model_id": "Qwen/Qwen3-8B"}).json()
    second = client.post("/ingest", json={"model_id": "Qwen/Qwen3-8B"}).json()
    assert first["checkpoints"][0]["card_revision"] == second["checkpoints"][0]["card_revision"]
    assert len(client.get("/models").json()) == 1


def test_ingest_accepts_a_full_url(client):
    """R1.1 - a model ID or a full HF URL, without further user input."""
    r = client.post("/ingest", json={"model_id": "https://huggingface.co/Qwen/Qwen3-8B"})
    assert r.status_code == 201
    assert r.json()["model_id"] == "Qwen/Qwen3-8B"


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------


def test_models_list_is_empty_before_any_ingest(client):
    assert client.get("/models").json() == []


def test_models_list_after_ingests(client):
    client.post("/ingest", json={"model_id": "Qwen/Qwen3-8B"})
    client.post("/ingest", json={"model_id": "nvidia/Nemotron-H-8B-Base-8K"})
    ids = {m["model_id"] for m in client.get("/models").json()}
    assert ids == {"Qwen/Qwen3-8B", "nvidia/Nemotron-H-8B-Base-8K"}


def test_unknown_model_is_404(client):
    assert client.get("/models/who/knows").status_code == 404


def test_detail_view_includes_null_fields(client):
    """R6.3 - null is information and must not be hidden."""
    client.post("/ingest", json={"model_id": "Qwen/Qwen3-8B-GGUF"})
    body = client.get("/models/Qwen/Qwen3-8B-GGUF").json()
    derived = body["checkpoints"][0]["derived"]
    assert "architecture" in derived
    assert derived["architecture"] is None
    assert derived["underivable"], "must name what it could not compute"


def test_gguf_model_exposes_a_checkpoint_per_quantization(client):
    client.post("/ingest", json={"model_id": "Qwen/Qwen3-8B-GGUF"})
    body = client.get("/models/Qwen/Qwen3-8B-GGUF").json()
    quants = {c["quantization"] for c in body["checkpoints"]}
    assert quants == {"Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0"}


def test_vram_estimate_is_served_with_its_assumptions(client):
    """R2.5 - an estimate without assumptions is not acceptable output."""
    client.post("/ingest", json={"model_id": "Qwen/Qwen3-8B"})
    vram = client.get("/models/Qwen/Qwen3-8B").json()["checkpoints"][0]["derived"]["vram"]
    assert vram["total_bytes"] is not None
    assert vram["assumptions"]["context"] > 0
    assert vram["assumptions"]["kv_dtype"]
    assert vram["assumptions"]["overhead_bytes"] > 0


def test_hybrid_model_is_served_as_hybrid(client):
    client.post("/ingest", json={"model_id": "nvidia/Nemotron-H-8B-Base-8K"})
    layers = client.get("/models/nvidia/Nemotron-H-8B-Base-8K").json()["checkpoints"][0]["derived"][
        "layers"
    ]
    assert layers["family"] == "nemotron_h"
    assert layers["recurrent"] == 24


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------


def test_benchmarks_list_is_empty_initially(client):
    assert client.get("/benchmarks").json() == []


def test_unknown_benchmark_is_404(client):
    assert client.get("/benchmarks/swe-bench-verified").status_code == 404


def test_benchmark_stub_is_created_and_marked_unwritten(client, vault):
    """R5.3 - a referenced benchmark with no document gets a stub, not silence."""
    vault.stub_benchmark("swe-bench-verified", referring_model="Qwen/Qwen3-8B")
    body = client.get("/benchmarks/swe-bench-verified").json()
    assert body["unwritten"] is True
    assert body["explanation"] == "", "R5.2 - never generated"
    assert "Qwen/Qwen3-8B" in body["referring_models"]


def test_benchmark_list_includes_stubs(client, vault):
    vault.stub_benchmark("swe-bench-verified", referring_model="a/b")
    vault.stub_benchmark("gpqa-diamond", referring_model="a/b")
    slugs = {b["slug"] for b in client.get("/benchmarks").json()}
    assert slugs == {"swe-bench-verified", "gpqa-diamond"}


# ---------------------------------------------------------------------------
# R7.2 - reads need no network
# ---------------------------------------------------------------------------


def test_reads_work_with_the_fetcher_removed(client, vault):
    client.post("/ingest", json={"model_id": "Qwen/Qwen3-8B"})

    def exploding_fetcher(_model_id):
        raise AssertionError("read paths must not touch the network")

    app.dependency_overrides[get_fetcher] = lambda: exploding_fetcher
    assert client.get("/models").status_code == 200
    assert client.get("/models/Qwen/Qwen3-8B").status_code == 200


# ---------------------------------------------------------------------------
# deleting a model
# ---------------------------------------------------------------------------


def test_delete_removes_the_model_and_returns_no_content(client, vault):
    vault.write(ModelDoc(model_id="a/one", checkpoints=[Checkpoint(repo="a/one")]), "ingest")

    response = client.delete("/models/a/one")

    assert response.status_code == 204
    assert vault.read("a/one") is None
    assert client.get("/models").json() == []


def test_deleting_a_model_that_is_not_there_is_a_404(client):
    assert client.delete("/models/nobody/nothing").status_code == 404


def test_delete_accepts_a_full_url_like_every_other_route(client, vault):
    """R1.1 - the same identifier handling ingest has, so a pasted URL works."""
    vault.write(ModelDoc(model_id="a/one", checkpoints=[Checkpoint(repo="a/one")]), "ingest")

    response = client.delete("/models/https://huggingface.co/a/one")

    assert response.status_code == 204
    assert vault.read("a/one") is None
