"""The one call that crosses the models/summary boundary.

A model entering the vault for the first time is summarised on the way in, so
nobody has to ask for the first one (see :func:`summarise_after_first_ingest`).
That behaviour needs both features -- models' ingest to know when "first time"
is true, summary's generator to write the block -- and putting it in either
feature's ``service.py`` would recreate the cycle this module exists to avoid.
Common is where two features' logic is allowed to meet on purpose.
"""

import logging

from app.core.config import LLMSettings, TavilySettings
from app.core.document import ModelDoc
from app.core.llm import LLMError
from app.core.search import SearchError
from app.core.store import Store
from app.features.summary.generate import generate_summary

log = logging.getLogger(__name__)


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
