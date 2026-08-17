"""One POST to an OpenAI-compatible endpoint (R7.4).

This module knows about HTTP and nothing else -- no prompts, no model cards, no
grounding. It is the only place that talks to the extraction endpoint, so every
way that call can fail is enumerated here, and each one raises rather than
returning something a caller might mistake for an answer.

The client is injectable for the same reason it is in ``backend.models.fetch``:
so the layers above can be tested with the network removed entirely.
"""

import json

import httpx

from backend.core.config import LLMSettings


class LLMError(RuntimeError):
    """The endpoint could not be used, or its answer could not be trusted."""


def complete(
    messages: list[dict],
    schema: dict,
    *,
    settings: LLMSettings,
    client: httpx.Client | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Send one chat completion constrained by ``schema`` and return the parsed object.

    ``max_tokens`` defaults to the configured budget, which is generous because
    the endpoint may front a reasoning model whose hidden reasoning consumes the
    same budget as the answer. A response cut short is rejected outright: half a
    JSON document repaired into something parseable is exactly the kind of
    plausible-looking output this project exists to not produce.
    """
    max_tokens = settings.max_tokens if max_tokens is None else max_tokens
    payload = {
        "model": settings.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "extraction", "strict": True, "schema": schema},
        },
    }
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    owned = client is None
    client = client or httpx.Client(timeout=settings.timeout)
    try:
        response = client.post(
            f"{settings.base_url}/chat/completions", json=payload, headers=headers
        )
    except httpx.HTTPError as exc:
        raise LLMError(f"extraction endpoint unreachable: {exc}") from exc
    finally:
        if owned:
            client.close()

    if response.status_code != 200:
        raise LLMError(
            f"extraction endpoint returned {response.status_code}: {response.text[:300]}"
        )

    choice = (response.json().get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length":
        raise LLMError(
            f"response was truncated at max_tokens={max_tokens}; "
            "a partial answer is not a partial result"
        )

    content = (choice.get("message") or {}).get("content") or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"response was not valid JSON: {content[:300]}") from exc
