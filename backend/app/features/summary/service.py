"""The summary feature's behaviour, independent of HTTP.

Summarisation is its own operation for the same reasons extraction is: it costs
tokens and a search call, the endpoints it depends on can be down, and neither
fact should decide whether a model can be ingested.
"""

from collections.abc import Callable

from app.common.exceptions import NotFound, Unprocessable
from app.core.config import LLMSettings, TavilySettings
from app.core.document import ModelDoc
from app.core.http import normalise_model_id
from app.core.store import Store
from app.features.summary.generate import generate_summary

CardFetcher = Callable[[str], tuple[str | None, str | None]]


def summarise_model(
    model_id: str,
    store: Store,
    *,
    fetch_card: CardFetcher,
    llm: LLMSettings,
    tavily: TavilySettings,
) -> ModelDoc:
    """Read the card, search the web, write an account of the model.

    Unlike extraction this replaces what was there: regenerating is the point of
    the button, and ``generated_on`` records which run produced the text on
    screen.
    """
    model_id = normalise_model_id(model_id)
    doc = store.read(model_id)
    if doc is None:
        raise NotFound(f"{model_id} is not in the store")

    card, _ = fetch_card(model_id)
    if not card:
        raise Unprocessable(f"{model_id} has no model card to read")

    derived = doc.checkpoints[0].derived if doc.checkpoints else None
    summary = generate_summary(model_id, card, derived, llm=llm, tavily=tavily)
    return store.merge_summary(model_id, summary)
