"""The extraction feature's HTTP surface.

Extraction is its own operation rather than part of ingest: a card refresh should
not cost tokens, the endpoint being down should not block ingest, and the block
this writes is not one ingest owns.
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.common.deps import StoreDep
from app.core.config import LLMSettings, llm_settings
from app.core.http import http_error, normalise_model_id
from app.core.schemas import ModelDoc
from app.extraction.extract import extract as extract_from_card
from app.extraction.llm import LLMError
from app.models.fetch import IngestError, fetch_snapshot

router = APIRouter(tags=["extraction"])


def get_card_fetcher():
    """Returns ``model_id -> (card_text, revision)``. Injected so tests stay offline."""

    def fetch_card(model_id: str) -> tuple[str | None, str | None]:
        snapshot = fetch_snapshot(model_id)
        return snapshot.readme, snapshot.revision

    return fetch_card


def get_llm_settings() -> LLMSettings:
    return llm_settings()


CardFetcher = Callable[[str], tuple[str | None, str | None]]
CardFetcherDep = Annotated[CardFetcher, Depends(get_card_fetcher)]
SettingsDep = Annotated[LLMSettings, Depends(get_llm_settings)]


@router.post("/models/{model_id:path}/extract", response_model=ModelDoc)
def extract_model(
    model_id: str,
    store: StoreDep,
    fetch_card: CardFetcherDep,
    settings: SettingsDep,
) -> ModelDoc:
    """Read the card, ask the model to locate facts in it, store what verifies.

    A run that verifies nothing is a success: most cards state none of this, and
    an empty result with its rejections is information (R3.2).
    """
    model_id = normalise_model_id(model_id)
    doc = store.read(model_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the store")
    if not doc.checkpoints:
        raise HTTPException(status_code=422, detail=f"{model_id} has no checkpoints to extract for")

    try:
        card, revision = fetch_card(model_id)
    except IngestError as exc:
        raise http_error(exc) from exc
    if not card:
        raise HTTPException(status_code=422, detail=f"{model_id} has no model card to read")

    try:
        extracted = extract_from_card(card, card_revision=revision or "", settings=settings)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    checkpoint = doc.checkpoints[0]
    return store.merge_extraction(
        model_id,
        repo=checkpoint.repo,
        quantization=checkpoint.quantization,
        extracted=extracted,
    )
