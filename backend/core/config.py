"""Where the service reads its configuration from.

Everything here is resolved from the environment so that nothing about a
deployment is baked into the code (R7.4 says the same about the extraction
endpoint). Keeping it in one module means the CLI, the API and the tests all
agree on where the vault is without importing each other.
"""

import os
from pathlib import Path

from pydantic import BaseModel

from backend.core.store import Store

# backend/core/config.py -> backend/ -> repo root. The vault is a sibling of the
# package, not a child of it (R7.6: it is a separate git repository).
DEFAULT_VAULT = Path(__file__).resolve().parents[2] / "vault"


def store_from_env() -> Store:
    """The vault named by ``LLMDEX_VAULT``, or the sibling ``./vault`` directory."""
    return Store(os.environ.get("LLMDEX_VAULT", DEFAULT_VAULT))


class LLMNotConfigured(RuntimeError):
    """No extraction endpoint is configured.

    There is no default and there must not be one: silently pointing at some
    other model would produce spans nobody can account for (R7.4).
    """


class LLMSettings(BaseModel):
    """R7.4 - any OpenAI-compatible endpoint, named by base URL and model."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 300.0
    """Extraction sends a whole model card and waits for a considered answer.
    Cards over 80k characters have taken 90 seconds against a reasoning model."""
    max_tokens: int = 32000
    """Output budget for one extraction.

    Generous on purpose. A reasoning model spends this budget on hidden
    reasoning before it writes a single field, and running out mid-document is
    a hard failure rather than a partial result -- the answer is thrown away
    (see :func:`backend.extraction.llm.complete`). 8000 was not enough for a
    135M model's card, which is a fact about how much the model thinks, not
    about how much there is to extract.
    """


class TavilyNotConfigured(RuntimeError):
    """No search key is configured.

    No default, for the same reason :class:`LLMNotConfigured` has none:
    summarising from the card alone is a different feature than the one asked
    for, and silently degrading into it would hide that the search half never
    ran.
    """


class TavilySettings(BaseModel):
    """The web-search half of summarisation. A key is all Tavily needs."""

    api_key: str
    timeout: float = 30.0
    max_results: int = 5
    """Enough context for a paragraph about the model without paying for a
    research session. One search per generation, not an agentic loop."""


def tavily_settings() -> TavilySettings:
    api_key = os.environ.get("LLMDEX_TAVILY_API_KEY", "").strip()
    if not api_key:
        raise TavilyNotConfigured("set LLMDEX_TAVILY_API_KEY to use summarisation")
    return TavilySettings(api_key=api_key)


def llm_settings() -> LLMSettings:
    base_url = os.environ.get("LLMDEX_LLM_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("LLMDEX_LLM_MODEL", "").strip()
    if not base_url or not model:
        raise LLMNotConfigured(
            "set LLMDEX_LLM_BASE_URL and LLMDEX_LLM_MODEL to use extraction "
            "(for example https://host/v1 and vllm/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4)"
        )
    return LLMSettings(
        base_url=base_url,
        model=model,
        api_key=os.environ.get("LLMDEX_LLM_API_KEY") or None,
        **_optional_int("LLMDEX_LLM_MAX_TOKENS", "max_tokens"),
        **_optional_int("LLMDEX_LLM_TIMEOUT", "timeout"),
    )


def _optional_int(variable: str, field: str) -> dict[str, int]:
    """Let a deployment override a budget without editing code (R7.4).

    An unset variable leaves the model's own default in place; a malformed one
    is an error rather than a silent fallback to the default, because a
    deployment that meant to raise the budget should hear that it did not.
    """
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return {}
    try:
        return {field: int(raw)}
    except ValueError as exc:
        raise LLMNotConfigured(f"{variable}={raw!r} is not an integer") from exc
