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


RECEIVED: list[list] = []
"""Every message list the scripted model was handed, newest last.

Recorded so a test can assert what actually reached the model, rather than
inferring it from what came back. `test_a_second_turn_sees_the_first_turn`
is the only reader; it clears this first.
"""


def scripted(messages, info):
    """Call the tool the first time, answer from its result the second."""
    RECEIVED.append(messages)
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

    Text is yielded whole, in one chunk. That is the shape that used to lose
    the answer: pydantic-ai folds a text part's opening chunk into its
    `PartStartEvent` and emits deltas only from the second chunk on, and
    `Frames` forwarded the deltas alone. An earlier version of this file split
    the text on spaces to keep a `content` frame flowing, which made the test
    pass over a bug that dropped the first word of every real answer.
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
            yield part.content


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

    # Both kinds: a part's opening chunk rides on `part_start` and the rest
    # arrives as `content`. Reading only one of them is how the dropped first
    # chunk went unnoticed.
    text = "".join(
        payload.get("text", "")
        for kind, payload in _frames(response.text)
        if kind in ("content", "part_start")
    )
    assert "It uses NVFP4." in text, "the answer arrived incomplete"


def test_the_run_ends_with_the_transcript(client):
    """R9.4 - the client appends this and sends it back next turn."""
    response = client.post("/models/a/one/chat", json={"message": "hi", "history": []})

    final = [p for kind, p in _frames(response.text) if p.get("phase") == "finished"]
    assert final, "the run must end with a finished frame"
    assert isinstance(final[0]["messages"], list)
    assert final[0]["messages"], "the transcript must not be empty"


def test_a_second_turn_sees_the_first_turn(client):
    """R9.4 - the round trip, not just the frame's shape.

    The test above only asserts the closing frame carries a non-empty list. It
    would pass just as happily if that list were the wrong messages, or only
    the newest ones -- and every conversation would silently start from nothing
    on turn two, which no single-turn test can see. This posts the transcript
    back the way the client will and asserts the model actually receives it.
    """
    first = client.post("/models/a/one/chat", json={"message": "quantization?", "history": []})
    transcript = next(p for k, p in _frames(first.text) if p.get("phase") == "finished")["messages"]

    RECEIVED.clear()
    second = client.post(
        "/models/a/one/chat",
        json={"message": "and the vLLM version?", "history": transcript},
    )

    assert second.status_code == 200
    assert RECEIVED, "the model was never called on the second turn"

    delivered = "\n".join(
        str(getattr(part, "content", "")) for message in RECEIVED[0] for part in message.parts
    )
    assert "quantization?" in delivered, "turn one's question did not survive the round trip"
    assert "NVFP4" in delivered, "turn one's tool result did not survive the round trip"
    assert "and the vLLM version?" in delivered, "the new question never reached the model"


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
