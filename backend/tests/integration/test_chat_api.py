"""The chat endpoint: what it streams, and how it refuses.

FunctionModel rather than a fixture: what is being tested is the loop -- that a
tool call is executed, its result fed back, and both reach the client -- not
that a particular model says a particular thing.
"""

import json
import subprocess

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from app.common.deps import get_card_fetcher, get_llm_settings, get_tavily_settings
from app.core.config import LLMSettings, TavilySettings
from app.core.document import Checkpoint, ModelDoc
from app.core.store import Store
from app.features.chat import router as chat_router
from app.main import app, get_store

CARD = "# One\n\n## Training Methodology\n\nQuantized to NVFP4 via PTQ.\n"
LLM = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model")
TAVILY = TavilySettings(api_key="tvly-test")


def scripted(messages, info):
    """Call the tool the first time, answer from its result the second."""
    if not any(getattr(part, "tool_name", None) for msg in messages for part in msg.parts):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="read_card_section",
                    args={"section": "Training Methodology"},
                    tool_call_id="c1",
                )
            ]
        )
    return ModelResponse(parts=[TextPart(content="It uses NVFP4.")])


async def scripted_stream(messages, info):
    """`run_stream_events` always calls the model's streaming path, so
    `FunctionModel` needs a `stream_function` even though nothing here cares
    about incremental deltas -- built from `scripted`.

    Text is split on spaces rather than yielded whole: pydantic-ai's parts
    manager folds a text part's first chunk into its `PartStartEvent` and only
    emits a `PartDeltaEvent` -- what `Frames` reports as `content` -- from the
    second chunk on, the same as a real token-streaming API's first chunk
    carrying only the role. One chunk would make the answer invisible to the
    `content`-kind assertion below for a reason that has nothing to do with
    the endpoint.
    """
    response = scripted(messages, info)
    for part in response.parts:
        if isinstance(part, ToolCallPart):
            yield {
                0: DeltaToolCall(
                    name=part.tool_name,
                    json_args=json.dumps(part.args),
                    tool_call_id=part.tool_call_id,
                )
            }
        elif isinstance(part, TextPart):
            words = part.content.split(" ")
            for i, word in enumerate(words):
                yield word if i == 0 else " " + word


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
def client(vault, monkeypatch):
    real_build = chat_router.build_agent
    # Entered but never exited, same as pydantic-ai's own pytest-fixture pattern
    # for overriding a model. The reference must be kept: `.override(...).__enter__()`
    # chained with nothing holding the returned context manager is collected by
    # CPython's refcounting GC immediately after the call, which runs the
    # generator's `finally:` and resets the override before the request that
    # needs it ever runs.
    live_overrides = []

    def build_with_test_model(llm, tavily, chat):
        agent = real_build(llm, tavily, chat)
        override = agent.override(model=FunctionModel(scripted, stream_function=scripted_stream))
        override.__enter__()
        live_overrides.append(override)
        return agent

    monkeypatch.setattr(chat_router, "build_agent", build_with_test_model)
    app.dependency_overrides[get_store] = lambda: vault
    app.dependency_overrides[get_card_fetcher] = lambda: lambda model_id: (CARD, "abc123")
    app.dependency_overrides[get_llm_settings] = lambda: LLM
    app.dependency_overrides[get_tavily_settings] = lambda: TAVILY
    yield TestClient(app)
    app.dependency_overrides.clear()
    for override in live_overrides:
        try:
            override.__exit__(None, None, None)
        except ValueError:
            # The token was set inside TestClient's own request-handling context
            # (a different `contextvars.Context` than this fixture's), so it
            # cannot be reset from here. Closing explicitly, rather than letting
            # GC do it later, is only to keep pytest's unraisable-exception
            # check quiet -- there is nothing left to undo either way.
            pass


def _frames(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


def test_a_tool_call_and_its_result_both_reach_the_client(client):
    """R9.6 - a tool call is visible, not hidden behind the answer."""
    response = client.post("/models/a/one/chat", json={"message": "quantization?", "history": []})

    assert response.status_code == 200
    kinds = [kind for kind, _ in _frames(response.text)]
    assert "tool_call" in kinds
    assert "tool_result" in kinds


def test_the_answer_is_streamed_as_content(client):
    response = client.post("/models/a/one/chat", json={"message": "quantization?", "history": []})

    text = "".join(
        payload.get("text", "") for kind, payload in _frames(response.text) if kind == "content"
    )
    assert "NVFP4" in text


def test_the_run_ends_with_the_transcript(client):
    """R9.4 - the client appends this and sends it back next turn."""
    response = client.post("/models/a/one/chat", json={"message": "hi", "history": []})

    final = [p for kind, p in _frames(response.text) if p.get("phase") == "finished"]
    assert final, "the run must end with a finished frame"
    assert isinstance(final[0]["messages"], list)
    assert final[0]["messages"], "the transcript must not be empty"


def test_an_unknown_model_is_refused_before_the_stream_opens(client):
    """Once the body has begun a 404 can no longer be sent."""
    response = client.post("/models/x/nope/chat", json={"message": "hi", "history": []})

    assert response.status_code == 404


def test_no_llm_configured_is_a_503(client, vault):
    from app.core.config import LLMNotConfigured

    def refuse():
        raise LLMNotConfigured("nothing configured")

    app.dependency_overrides[get_llm_settings] = refuse
    response = client.post("/models/a/one/chat", json={"message": "hi", "history": []})

    assert response.status_code == 503
