"""The OpenAI-compatible client (R7.4).

Configured entirely by environment so any endpoint can be pointed at. The tests
drive it through a stub transport rather than a live server: what matters here is
the request we build and which responses we refuse.
"""

import httpx
import pytest

from backend.core.config import LLMNotConfigured, LLMSettings, llm_settings
from backend.extraction.llm import LLMError, complete

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
SETTINGS = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model", api_key="k")


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLMDEX_LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLMDEX_LLM_MODEL", "vllm/some-model")
    monkeypatch.setenv("LLMDEX_LLM_API_KEY", "secret")

    settings = llm_settings()

    assert settings.base_url == "https://example.test/v1"
    assert settings.model == "vllm/some-model"


def test_an_unconfigured_endpoint_fails_loudly(monkeypatch):
    """No default endpoint. A missing configuration is not something to guess at."""
    monkeypatch.delenv("LLMDEX_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLMDEX_LLM_MODEL", raising=False)

    with pytest.raises(LLMNotConfigured):
        llm_settings()


def test_the_request_carries_the_model_schema_and_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}]},
        )

    with _client(handler) as client:
        result = complete(
            [{"role": "user", "content": "hi"}], SCHEMA, settings=SETTINGS, client=client
        )

    assert result == {"ok": True}
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "vllm/some-model"
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True


def test_a_truncated_response_is_an_error_not_a_partial_result():
    """A reasoning model burns output budget on `reasoning`; truncated JSON is a failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "length", "message": {"content": '{"ok": tr'}}]},
        )

    with _client(handler) as client, pytest.raises(LLMError, match="truncated"):
        complete([{"role": "user", "content": "hi"}], SCHEMA, settings=SETTINGS, client=client)


def test_unparseable_content_is_never_repaired():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "sorry, no"}}]},
        )

    with _client(handler) as client, pytest.raises(LLMError, match="not valid JSON"):
        complete([{"role": "user", "content": "hi"}], SCHEMA, settings=SETTINGS, client=client)


def test_an_http_error_names_the_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "provider is required"}})

    with _client(handler) as client, pytest.raises(LLMError, match="400"):
        complete([{"role": "user", "content": "hi"}], SCHEMA, settings=SETTINGS, client=client)
