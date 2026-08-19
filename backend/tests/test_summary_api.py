"""The summarise endpoint, and the one ingest triggers by itself.

Summarisation reaches the network three times - card, search, completion - so all
three are injected and none is touched here.
"""

import subprocess

import pytest
from fastapi.testclient import TestClient

from app.core.config import LLMSettings, TavilyNotConfigured, TavilySettings
from app.core.llm import LLMError
from app.core.schemas import Checkpoint, ModelDoc, Summary
from app.core.search import SearchError
from app.core.store import Store
from app.features.summary.router import (
    get_card_fetcher,
    get_llm_settings,
    get_optional_llm_settings,
    get_optional_tavily_settings,
    get_tavily_settings,
)
from app.main import app, get_store

CARD = "# One\n\nA small model for testing.\n"
LLM = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model")
TAVILY = TavilySettings(api_key="tvly-test")

ANSWER = {
    "overview": "One is a small model.",
    "unique_points": ["Tiny"],
    "pros": ["Fast"],
    "cons": ["Small"],
    "use_cases": ["Testing"],
}


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
    app.dependency_overrides[get_llm_settings] = lambda: LLM
    app.dependency_overrides[get_tavily_settings] = lambda: TAVILY
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def endpoints(monkeypatch):
    """The real generate runs; only its two network calls are stubbed."""
    monkeypatch.setattr("app.features.summary.generate.complete", lambda *a, **kw: ANSWER)
    monkeypatch.setattr("app.features.summary.generate.search", lambda *a, **kw: [])


def test_summarising_writes_the_block(client, endpoints):
    response = client.post("/models/a/one/summarize")

    assert response.status_code == 200, response.text
    summary = response.json()["summary"]
    assert summary["overview"] == "One is a small model."
    assert summary["cons"] == ["Small"]
    assert summary["generated_by"] == "vllm/some-model"


def test_summarising_an_unknown_model_is_a_404(client, endpoints):
    assert client.post("/models/nobody/nothing/summarize").status_code == 404


def test_a_failed_completion_is_a_502_and_writes_nothing(client, vault, monkeypatch):
    monkeypatch.setattr("app.features.summary.generate.search", lambda *a, **kw: [])
    monkeypatch.setattr(
        "app.features.summary.generate.complete",
        lambda *a, **kw: (_ for _ in ()).throw(LLMError("unreachable")),
    )

    response = client.post("/models/a/one/summarize")

    assert response.status_code == 502
    assert vault.read("a/one").summary is None


def test_a_failed_search_is_a_502_and_writes_nothing(client, vault, monkeypatch):
    monkeypatch.setattr(
        "app.features.summary.generate.search",
        lambda *a, **kw: (_ for _ in ()).throw(SearchError("search endpoint returned 401")),
    )

    response = client.post("/models/a/one/summarize")

    assert response.status_code == 502
    assert vault.read("a/one").summary is None


def test_an_unconfigured_endpoint_is_a_503(vault):
    app.dependency_overrides[get_store] = lambda: vault
    app.dependency_overrides[get_card_fetcher] = lambda: lambda model_id: (CARD, "abc123")

    def unconfigured():
        raise TavilyNotConfigured("set LLMDEX_TAVILY_API_KEY")

    app.dependency_overrides[get_llm_settings] = lambda: LLM
    app.dependency_overrides[get_tavily_settings] = unconfigured
    try:
        response = TestClient(app).post("/models/a/one/summarize")
        assert response.status_code == 503
        assert "LLMDEX_TAVILY_API_KEY" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_regenerating_replaces_the_previous_summary(client, vault, endpoints):
    vault.merge_summary(
        "a/one",
        Summary(overview="Stale.", generated_by="old", generated_on="2026-01-01"),
    )

    response = client.post("/models/a/one/summarize")

    assert response.json()["summary"]["overview"] == "One is a small model."


# ---------------------------------------------------------------------------
# the first ingest generates one by itself; later ones never do
# ---------------------------------------------------------------------------


def test_a_first_ingest_generates_a_summary(vault, monkeypatch, endpoints):
    """The user should not have to ask for the first one."""
    from app.features.models.router import get_fetcher
    from tests.test_api import snapshot

    app.dependency_overrides[get_store] = lambda: vault
    app.dependency_overrides[get_fetcher] = lambda: lambda model_id: snapshot("qwen3-8b")
    app.dependency_overrides[get_optional_llm_settings] = lambda: LLM
    app.dependency_overrides[get_optional_tavily_settings] = lambda: TAVILY
    try:
        response = TestClient(app).post("/ingest", json={"model_id": "Qwen/Qwen3-8B"})

        assert response.status_code == 201, response.text
        assert response.json()["summary"]["overview"] == "One is a small model."
    finally:
        app.dependency_overrides.clear()


def test_a_re_ingest_does_not_generate_again(vault, monkeypatch, endpoints):
    """A card refresh must not silently spend tokens, and must not overwrite a
    summary somebody regenerated on purpose."""
    from app.features.models.router import get_fetcher
    from tests.test_api import snapshot

    calls = []
    monkeypatch.setattr(
        "app.features.summary.generate.complete", lambda *a, **kw: calls.append(1) or ANSWER
    )

    app.dependency_overrides[get_store] = lambda: vault
    app.dependency_overrides[get_fetcher] = lambda: lambda model_id: snapshot("qwen3-8b")
    app.dependency_overrides[get_optional_llm_settings] = lambda: LLM
    app.dependency_overrides[get_optional_tavily_settings] = lambda: TAVILY
    try:
        client = TestClient(app)
        client.post("/ingest", json={"model_id": "Qwen/Qwen3-8B"})
        client.post("/ingest", json={"model_id": "Qwen/Qwen3-8B"})

        assert len(calls) == 1
    finally:
        app.dependency_overrides.clear()


def test_ingest_survives_a_summary_that_cannot_be_generated(vault, monkeypatch):
    """R1.5 - ingest is atomic and owns the document. Whether some other endpoint
    was reachable is not allowed to decide if a model can enter the vault."""
    from app.features.models.router import get_fetcher
    from tests.test_api import snapshot

    monkeypatch.setattr(
        "app.features.summary.generate.search",
        lambda *a, **kw: (_ for _ in ()).throw(SearchError("unreachable")),
    )

    app.dependency_overrides[get_store] = lambda: vault
    app.dependency_overrides[get_fetcher] = lambda: lambda model_id: snapshot("qwen3-8b")
    app.dependency_overrides[get_optional_llm_settings] = lambda: LLM
    app.dependency_overrides[get_optional_tavily_settings] = lambda: TAVILY
    try:
        response = TestClient(app).post("/ingest", json={"model_id": "Qwen/Qwen3-8B"})

        assert response.status_code == 201, response.text
        assert response.json()["summary"] is None
    finally:
        app.dependency_overrides.clear()


def test_ingest_survives_summarisation_being_unconfigured(vault, monkeypatch):
    """A deployment with no search key still ingests; it just has no summaries."""
    from app.features.models.router import get_fetcher
    from tests.test_api import snapshot

    monkeypatch.delenv("LLMDEX_TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("LLMDEX_LLM_BASE_URL", raising=False)

    app.dependency_overrides[get_store] = lambda: vault
    app.dependency_overrides[get_fetcher] = lambda: lambda model_id: snapshot("qwen3-8b")
    try:
        response = TestClient(app).post("/ingest", json={"model_id": "Qwen/Qwen3-8B"})

        assert response.status_code == 201, response.text
        assert response.json()["summary"] is None
    finally:
        app.dependency_overrides.clear()
