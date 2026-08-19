"""The extraction feature's behaviour, independent of HTTP.

Extraction is its own operation rather than part of ingest: a card refresh
should not cost tokens, the endpoint being down should not block ingest, and
the block this writes is not one ingest owns.
"""

from collections.abc import Callable

from app.common.exceptions import NotFound, Unprocessable
from app.core.config import LLMSettings
from app.core.document import ModelDoc
from app.core.http import normalise_model_id
from app.core.store import Store
from app.features.extraction.extract import extract as extract_from_card

CardFetcher = Callable[[str], tuple[str | None, str | None]]


def extract_model(
    model_id: str,
    store: Store,
    *,
    fetch_card: CardFetcher,
    settings: LLMSettings,
) -> ModelDoc:
    """Read the card, ask the model to locate facts in it, store what verifies.

    A run that verifies nothing is a success: most cards state none of this, and
    an empty result with its rejections is information (R3.2).
    """
    model_id = normalise_model_id(model_id)
    doc = store.read(model_id)
    if doc is None:
        raise NotFound(f"{model_id} is not in the store")
    if not doc.checkpoints:
        raise Unprocessable(f"{model_id} has no checkpoints to extract for")

    card, revision = fetch_card(model_id)
    if not card:
        raise Unprocessable(f"{model_id} has no model card to read")

    extracted = extract_from_card(card, card_revision=revision or "", settings=settings)

    checkpoint = doc.checkpoints[0]
    return store.merge_extraction(
        model_id,
        repo=checkpoint.repo,
        quantization=checkpoint.quantization,
        extracted=extracted,
    )
