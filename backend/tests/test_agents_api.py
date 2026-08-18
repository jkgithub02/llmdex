"""The stream endpoint: what it emits, and how it refuses."""

import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from backend.agents import runner
from backend.agents.events import AgentEvent
from backend.core.config import LLMSettings, TavilySettings
from backend.core.schemas import Checkpoint, ModelDoc
from backend.core.store import Store
from backend.main import app, get_store
from backend.summary.router import get_card_fetcher, get_llm_settings, get_tavily_settings

CARD = "# One\n\nA small model.\n"
LLM = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model")
TAVILY = TavilySettings(api_key="tvly-test")


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    store = Store(root)
    store.write(
        ModelDoc(model_id="a/one", checkpoints=[Checkpoint(repo="a/one")]), operation="ingest"
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
def fake_agent(monkeypatch):
    def agent(doc, card, *, llm, tavily, store, emit, card_revision=None):
        emit(AgentEvent(agent="about", kind="reasoning", text="line one\nline two"))
        return "summary"

    monkeypatch.setattr(runner, "AGENTS", {"about": agent})


def test_the_stream_is_server_sent_events(client, fake_agent):
    with client.stream("GET", "/models/a/one/agents/stream?agents=about") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: phase" in body
    assert "event: reasoning" in body
    assert '"phase": "started"' in body or '"phase":"started"' in body


def test_reasoning_with_newlines_survives_the_wire(client, fake_agent):
    """The fake agent emits "line one\nline two". A raw newline inside a data:
    line would end the frame early and leave an unparseable fragment, so each
    frame must carry exactly one data: line of complete JSON."""
    with client.stream("GET", "/models/a/one/agents/stream?agents=about") as response:
        body = "".join(response.iter_text())

    frames = [frame for frame in body.split("\n\n") if frame.strip()]
    reasoning = [frame for frame in frames if frame.startswith("event: reasoning")]
    assert reasoning, "no reasoning frame arrived"

    for frame in reasoning:
        data_lines = [line for line in frame.split("\n") if line.startswith("data: ")]
        assert len(data_lines) == 1, f"frame split across lines: {frame!r}"
        payload = json.loads(data_lines[0].removeprefix("data: "))
        assert "\n" in payload["text"], "the newline must survive inside the JSON payload"


def test_streaming_for_an_unknown_model_is_a_404(client, fake_agent):
    response = client.get("/models/nobody/nothing/agents/stream?agents=about")
    assert response.status_code == 404


def test_an_unknown_agent_name_is_a_422(client, fake_agent):
    response = client.get("/models/a/one/agents/stream?agents=wizard")
    assert response.status_code == 422
    assert "wizard" in response.json()["detail"]


def test_a_model_with_no_card_is_a_422(vault, fake_agent):
    app.dependency_overrides[get_store] = lambda: vault
    app.dependency_overrides[get_card_fetcher] = lambda: lambda model_id: (None, None)
    app.dependency_overrides[get_llm_settings] = lambda: LLM
    app.dependency_overrides[get_tavily_settings] = lambda: TAVILY
    try:
        response = TestClient(app).get("/models/a/one/agents/stream?agents=about")
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()
