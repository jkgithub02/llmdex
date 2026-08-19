"""The Tavily search client.

Same shape as the LLM client tests: a stub transport, because what matters here
is the request we build and which responses we refuse. The response shape was
confirmed against the live API before this was written.
"""

import httpx
import pytest

from app.core.config import TavilyNotConfigured, TavilySettings, tavily_settings
from app.search.tavily import SearchError, search

SETTINGS = TavilySettings(api_key="tvly-test")


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLMDEX_TAVILY_API_KEY", "tvly-secret")

    assert tavily_settings().api_key == "tvly-secret"


def test_an_unconfigured_key_fails_loudly(monkeypatch):
    """No default. Summarising without the search half is a different feature."""
    monkeypatch.delenv("LLMDEX_TAVILY_API_KEY", raising=False)

    with pytest.raises(TavilyNotConfigured, match="LLMDEX_TAVILY_API_KEY"):
        tavily_settings()


def test_the_request_carries_the_query_and_bearer_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://example.test/a", "title": "A", "content": "about the model"}
                ]
            },
        )

    with _client(handler) as client:
        results = search("Qwen3-8B model", settings=SETTINGS, client=client)

    assert [r.url for r in results] == ["https://example.test/a"]
    assert results[0].content == "about the model"
    assert seen["url"] == "https://api.tavily.com/search"
    assert seen["auth"] == "Bearer tvly-test"
    assert seen["body"]["query"] == "Qwen3-8B model"


def test_no_results_is_not_an_error():
    """A model with no web presence is a fact, not a failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    with _client(handler) as client:
        assert search("nobody/nothing", settings=SETTINGS, client=client) == []


def test_an_http_error_names_the_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "unauthorized"})

    with _client(handler) as client, pytest.raises(SearchError, match="401"):
        search("anything", settings=SETTINGS, client=client)


def test_an_unreachable_endpoint_is_a_search_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with _client(handler) as client, pytest.raises(SearchError, match="unreachable"):
        search("anything", settings=SETTINGS, client=client)
