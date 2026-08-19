"""The summary feature's HTTP surface.

Summarisation is its own operation for the same reasons extraction is: it costs
tokens and a search call, the endpoints it depends on can be down, and neither
fact should decide whether a model can be ingested.

The one exception is the first ingest of a model, which generates a summary
without being asked -- see :func:`summarise_after_first_ingest`.
"""

import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.common.deps import StoreDep
from app.core.config import (
    LLMNotConfigured,
    LLMSettings,
    TavilyNotConfigured,
    TavilySettings,
    llm_settings,
    tavily_settings,
)
from app.core.document import ModelDoc
from app.core.http import http_error, normalise_model_id
from app.core.llm import LLMError
from app.core.search import SearchError
from app.core.store import Store
from app.features.models.fetch import IngestError, fetch_snapshot
from app.features.summary.generate import generate_summary

log = logging.getLogger(__name__)

router = APIRouter(tags=["summary"])


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

    The summarise endpoint should say 503 when nothing is configured -- the
    caller asked for a summary and cannot have one. Ingest must not: a vault
    with no LLM configured is a perfectly good vault, it just has no summaries.
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


CardFetcher = Callable[[str], tuple[str | None, str | None]]
CardFetcherDep = Annotated[CardFetcher, Depends(get_card_fetcher)]
LLMDep = Annotated[LLMSettings, Depends(get_llm_settings)]
TavilyDep = Annotated[TavilySettings, Depends(get_tavily_settings)]
OptionalLLMDep = Annotated[LLMSettings | None, Depends(get_optional_llm_settings)]
OptionalTavilyDep = Annotated[TavilySettings | None, Depends(get_optional_tavily_settings)]


@router.post("/models/{model_id:path}/summarize", response_model=ModelDoc)
def summarise_model(
    model_id: str,
    store: StoreDep,
    fetch_card: CardFetcherDep,
    llm: LLMDep,
    tavily: TavilyDep,
) -> ModelDoc:
    """Read the card, search the web, write an account of the model.

    Unlike extraction this replaces what was there: regenerating is the point of
    the button, and ``generated_on`` records which run produced the text on
    screen.
    """
    model_id = normalise_model_id(model_id)
    doc = store.read(model_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the store")

    try:
        card, _ = fetch_card(model_id)
    except IngestError as exc:
        raise http_error(exc) from exc
    if not card:
        raise HTTPException(status_code=422, detail=f"{model_id} has no model card to read")

    derived = doc.checkpoints[0].derived if doc.checkpoints else None

    try:
        summary = generate_summary(model_id, card, derived, llm=llm, tavily=tavily)
    except (LLMError, SearchError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return store.merge_summary(model_id, summary)


def summarise_after_first_ingest(
    doc: ModelDoc,
    store: Store,
    card: str | None,
    *,
    llm: LLMSettings | None,
    tavily: TavilySettings | None,
) -> ModelDoc:
    """Generate a summary for a model nobody has summarised yet, and never fail.

    This is the one place in the codebase that swallows an error, and it is
    deliberate. Ingest is atomic and owns the document (R1.5); whether a search
    API answered is not allowed to decide whether a model can enter the vault.
    The failure is not hidden from anyone either -- the document comes back with
    no summary, which the UI renders as "not generated yet, press the button",
    the same state as a model nobody has asked for a summary of.

    Only the first time. A re-ingest must not spend tokens refreshing a card, and
    must not overwrite a summary somebody regenerated on purpose.

    The card is passed in because ingest already holds it; fetching it again here
    would pay for a second request to answer a question already answered.
    """
    if doc.summary is not None or not card or llm is None or tavily is None:
        return doc

    try:
        summary = generate_summary(
            doc.model_id,
            card,
            doc.checkpoints[0].derived if doc.checkpoints else None,
            llm=llm,
            tavily=tavily,
        )
    except (LLMError, SearchError) as exc:
        log.warning("could not summarise %s on ingest: %s", doc.model_id, exc)
        return doc

    return store.merge_summary(doc.model_id, summary)
