"""One POST to Tavily's search API.

This module knows about HTTP and nothing else -- no prompts, no model cards, no
summaries. It is the only place that talks to the search endpoint, so every way
that call can fail is enumerated here, and each one raises rather than returning
something a caller might mistake for an answer.

The client is injectable for the same reason it is in
``app.extraction.llm``: so the layers above can be tested with the network
removed entirely.
"""

import httpx
from pydantic import BaseModel

from app.core.config import TavilySettings

ENDPOINT = "https://api.tavily.com/search"


class SearchError(RuntimeError):
    """The search endpoint could not be used."""


class SearchResult(BaseModel):
    """One page, as Tavily summarised it.

    ``url`` is kept because it is what a reader follows to check a claim the
    card did not support -- the same job ``source_url`` does for a benchmark
    score (R4.5a).
    """

    url: str
    title: str = ""
    content: str = ""


def search(
    query: str,
    *,
    settings: TavilySettings,
    client: httpx.Client | None = None,
) -> list[SearchResult]:
    """Search the web for ``query``.

    An empty result list is a legitimate answer: a model nobody has written
    about is a fact about the model, not a failure of the search.
    """
    payload = {"query": query, "max_results": settings.max_results}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.api_key}",
    }

    owned = client is None
    client = client or httpx.Client(timeout=settings.timeout)
    try:
        response = client.post(ENDPOINT, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise SearchError(f"search endpoint unreachable: {exc}") from exc
    finally:
        if owned:
            client.close()

    if response.status_code != 200:
        raise SearchError(f"search endpoint returned {response.status_code}: {response.text[:300]}")

    try:
        results = response.json().get("results") or []
    except ValueError as exc:
        raise SearchError(f"search response was not valid JSON: {response.text[:300]}") from exc

    return [
        SearchResult(
            url=row.get("url") or "",
            title=row.get("title") or "",
            content=row.get("content") or "",
        )
        for row in results
        if row.get("url")
    ]
