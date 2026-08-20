"""FastAPI dependencies shared by more than one feature router.

``get_store`` is a function rather than a module-level value so tests can replace
it through ``app.dependency_overrides`` -- which is how the read endpoints are
exercised with the network removed entirely (R7.2).

A dependency only one feature uses lives with that feature. These do not: the
card fetcher and the LLM/Tavily settings providers are wired into the
extraction, summary and agents routers alike, and previously lived in
whichever router happened to define them first, with the others importing it
from there -- a feature reaching into a sibling's router to borrow its
``Depends``. That is a cycle wearing a DI costume, so the providers live here
instead, where any feature may import them without importing each other.

This is also the one place a "provider" is allowed to depend on a feature
(``fetch_snapshot`` from ``models.fetch``): ``core`` must never import a
feature, but ``common`` gluing two features' DI together for the sake of
breaking a cycle between them is exactly what it is for.
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends

from app.core.config import (
    ChatSettings,
    LLMNotConfigured,
    LLMSettings,
    TavilyNotConfigured,
    TavilySettings,
    chat_settings,
    llm_settings,
    store_from_env,
    tavily_settings,
)
from app.core.store import Store
from app.features.models.fetch import fetch_snapshot


def get_store() -> Store:
    return store_from_env()


StoreDep = Annotated[Store, Depends(get_store)]


def get_card_fetcher():
    """Returns ``model_id -> (card_text, revision)``. Injected so tests stay offline."""

    def fetch_card(model_id: str) -> tuple[str | None, str | None]:
        snapshot = fetch_snapshot(model_id)
        return snapshot.readme, snapshot.revision

    return fetch_card


def get_llm_settings() -> LLMSettings:
    return llm_settings()


def get_tavily_settings() -> TavilySettings:
    return tavily_settings()


def get_optional_llm_settings() -> LLMSettings | None:
    """The same settings, but absent rather than fatal.

    The summarise and extract endpoints should say 503 when nothing is
    configured -- the caller asked for one and cannot have it. Ingest must
    not: a vault with no LLM configured is a perfectly good vault, it just has
    no summaries.
    """
    try:
        return llm_settings()
    except LLMNotConfigured:
        return None


def get_optional_tavily_settings() -> TavilySettings | None:
    try:
        return tavily_settings()
    except TavilyNotConfigured:
        return None


def get_chat_settings() -> ChatSettings:
    return chat_settings()


CardFetcher = Callable[[str], tuple[str | None, str | None]]
CardFetcherDep = Annotated[CardFetcher, Depends(get_card_fetcher)]
LLMDep = Annotated[LLMSettings, Depends(get_llm_settings)]
TavilyDep = Annotated[TavilySettings, Depends(get_tavily_settings)]
OptionalLLMDep = Annotated[LLMSettings | None, Depends(get_optional_llm_settings)]
OptionalTavilyDep = Annotated[TavilySettings | None, Depends(get_optional_tavily_settings)]
ChatSettingsDep = Annotated[ChatSettings, Depends(get_chat_settings)]
