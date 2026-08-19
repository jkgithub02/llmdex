"""The two agents that already existed, wrapped so they can narrate.

Neither adapter may change what its underlying step produces -- the point of
Phase A is the narration, not new behaviour.
"""

import json
import subprocess

import pytest

from app.agents import about, prose
from app.agents.events import AgentEvent
from app.core.config import LLMSettings, TavilySettings
from app.core.schemas import Checkpoint, ModelDoc
from app.core.store import Store

LLM = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model")
TAVILY = TavilySettings(api_key="tvly-test")
CARD = "# One\n\nQuantized to NVFP4. Requires vLLM 0.27.1 or newer.\n"


@pytest.fixture
def store(tmp_path):
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


def test_an_event_serialises_as_one_sse_frame():
    frame = AgentEvent(agent="about", kind="reasoning", text="thinking").to_sse()

    assert frame.startswith("event: reasoning\n")
    assert frame.endswith("\n\n"), "an SSE frame ends with a blank line"
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload == {"agent": "about", "text": "thinking"}


def test_an_event_with_a_newline_stays_one_frame():
    """Reasoning arrives with newlines in it, and a raw newline would split the
    frame and truncate the message."""
    frame = AgentEvent(agent="about", kind="reasoning", text="one\ntwo").to_sse()

    assert frame.count("data: ") == 1
    assert json.loads(frame.split("data: ", 1)[1])["text"] == "one\ntwo"


def test_the_about_agent_writes_a_summary_and_narrates(store, monkeypatch):
    events: list[AgentEvent] = []
    monkeypatch.setattr("app.agents.about.search", lambda *a, **kw: [])

    def stream_json(messages, schema, *, on_reasoning=None, **kw):
        on_reasoning("weighing the card")
        return {
            "overview": "One is a small model.",
            "unique_points": [],
            "pros": [],
            "cons": [],
            "use_cases": [],
        }

    monkeypatch.setattr("app.agents.about.stream_json", stream_json)

    wrote = about.run(
        store.read("a/one"), CARD, llm=LLM, tavily=TAVILY, store=store, emit=events.append
    )

    assert wrote == "summary"
    assert store.read("a/one").summary.overview == "One is a small model."
    assert [e.text for e in events if e.kind == "reasoning"] == ["weighing the card"]


def test_the_prose_agent_still_grounds_every_value(store, monkeypatch):
    """R3.1 - narration must not loosen the copy-only rule."""
    events: list[AgentEvent] = []

    def stream_json(messages, schema, *, on_reasoning=None, **kw):
        on_reasoning("looking for a quantization section")
        return {"quantization": {"format": "NVFP4", "method": "invented"}}

    monkeypatch.setattr("app.agents.prose.stream_json", stream_json)

    wrote = prose.run(
        store.read("a/one"), CARD, llm=LLM, tavily=TAVILY, store=store, emit=events.append
    )

    assert wrote == "extracted"
    extracted = store.read("a/one").checkpoints[0].extracted
    assert extracted.quantization.format.text == "NVFP4"
    assert extracted.quantization.method is None, "a value not in the card must not be stored"
    assert [r.proposed for r in extracted.rejected] == ["invented"]
    assert any(e.kind == "reasoning" for e in events)


def test_the_prose_agent_stamps_the_card_it_was_actually_given(store, monkeypatch):
    """R1.3 - the spans are offsets into the card in hand. Stamping them with the
    revision from an earlier ingest would describe them against a card they were
    never found in, and the UI prints that as the revision it read."""
    monkeypatch.setattr(
        "app.agents.prose.stream_json",
        lambda *a, **kw: {"quantization": {"format": "NVFP4"}},
    )

    prose.run(
        store.read("a/one"),
        CARD,
        llm=LLM,
        tavily=TAVILY,
        store=store,
        emit=lambda event: None,
        card_revision="fresh123",
    )

    assert store.read("a/one").checkpoints[0].extracted.card_revision == "fresh123"


def test_the_prose_agent_asks_again_when_nothing_comes_back(store, monkeypatch):
    """The retry extraction.extract carries, kept here because the UI now routes
    all extraction through this path rather than POST /extract."""
    calls = []

    def stream_json(messages, schema, *, on_reasoning=None, **kw):
        calls.append(1)
        return {} if len(calls) == 1 else {"quantization": {"format": "NVFP4"}}

    monkeypatch.setattr("app.agents.prose.stream_json", stream_json)

    prose.run(store.read("a/one"), CARD, llm=LLM, tavily=TAVILY, store=store, emit=lambda e: None)

    assert len(calls) == 2, "an empty answer should have been retried"
    assert store.read("a/one").checkpoints[0].extracted.quantization.format.text == "NVFP4"


def test_the_prose_agent_believes_two_empty_answers(store, monkeypatch):
    calls = []

    def stream_json(messages, schema, *, on_reasoning=None, **kw):
        calls.append(1)
        return {}

    monkeypatch.setattr("app.agents.prose.stream_json", stream_json)

    prose.run(store.read("a/one"), CARD, llm=LLM, tavily=TAVILY, store=store, emit=lambda e: None)

    assert len(calls) == 2, "two agreeing empties are believed, not asked a third time"
