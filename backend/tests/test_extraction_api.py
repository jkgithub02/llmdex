"""The extract endpoint: what it writes, and how it refuses.

Extraction reaches the network twice - once for the card, once for the model - so
both are injected and neither is touched here.
"""

import subprocess

import pytest
from fastapi.testclient import TestClient

from app.core.config import LLMNotConfigured, LLMSettings
from app.core.document import Checkpoint, ModelDoc
from app.core.llm import LLMError
from app.core.store import Store
from app.features.extraction.router import get_card_fetcher, get_llm_settings
from app.main import app, get_store

CARD = """# Model Card

## Quantization

The model is quantized to NVFP4 using PTQ via NVIDIA ModelOpt.
"""

SETTINGS = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model")


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    store = Store(root)
    store.write(
        ModelDoc(
            model_id="a/one",
            checkpoints=[Checkpoint(repo="a/one", card_revision="abc123", ingested="2026-08-17")],
        ),
        operation="ingest",
    )
    return store


@pytest.fixture
def client(vault):
    app.dependency_overrides[get_store] = lambda: vault
    app.dependency_overrides[get_card_fetcher] = lambda: lambda model_id: (CARD, "abc123")
    app.dependency_overrides[get_llm_settings] = lambda: SETTINGS
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_extraction_writes_verified_spans(client, monkeypatch):
    """Only the HTTP call is stubbed; the real extract and grounding run."""
    monkeypatch.setattr(
        "app.features.extraction.extract.complete",
        lambda messages, schema, **kw: {"quantization": {"format": "NVFP4"}},
    )

    response = client.post("/models/a/one/extract")

    assert response.status_code == 200, response.text
    block = response.json()["checkpoints"][0]["extracted"]
    assert block["quantization"]["format"]["text"] == "NVFP4"
    assert block["card_revision"] == "abc123"


def test_extracting_an_unknown_model_is_a_404(client):
    response = client.post("/models/nobody/nothing/extract")
    assert response.status_code == 404


def test_an_unreachable_endpoint_is_a_502_and_writes_nothing(client, vault, monkeypatch):
    monkeypatch.setattr(
        "app.features.extraction.extract.complete",
        lambda messages, schema, **kw: (_ for _ in ()).throw(LLMError("unreachable")),
    )

    response = client.post("/models/a/one/extract")

    assert response.status_code == 502
    assert vault.read("a/one").checkpoints[0].extracted is None


def test_an_unconfigured_endpoint_is_a_503(vault):
    app.dependency_overrides[get_store] = lambda: vault
    app.dependency_overrides[get_card_fetcher] = lambda: lambda model_id: (CARD, "abc123")

    def unconfigured():
        raise LLMNotConfigured("set LLMDEX_LLM_BASE_URL")

    app.dependency_overrides[get_llm_settings] = unconfigured
    try:
        response = TestClient(app).post("/models/a/one/extract")
        assert response.status_code == 503
        assert "LLMDEX_LLM_BASE_URL" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
