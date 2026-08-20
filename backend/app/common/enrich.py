"""The agents a model gets on the way in, so no tab starts empty.

R1.7 names extraction as an ingest phase -- "fetching, deriving, extracting,
writing ... extraction can take tens of seconds" -- but only the summary ever
ran here, so Details and Benchmarks stayed blank until somebody pressed a
button on each of them. A model entering the vault now gets every agent that
has never run for it.

This is the one place in ``common`` besides ``deps.py`` that may import a
feature, and it does so through ``agents/runner.py`` rather than reaching for
three feature modules directly: the runner is already the designated
composition root for "run several agents", already isolates one agent's failure
from the others, and already knows how to narrate them. Assembling that a second
way here would be two things to keep in step.
"""

import logging

from app.core.config import LLMSettings, TavilySettings
from app.core.document import ModelDoc
from app.core.store import Store
from app.features.agents.runner import run_agents

log = logging.getLogger(__name__)

SUMMARY = "about"
EXTRACTION = "prose"
BENCHMARKS = "benchmarks"


def _outstanding(doc: ModelDoc, tavily: TavilySettings | None) -> list[str]:
    """The agents that have never run for this model.

    R1.4: a re-ingest must not spend tokens refreshing a block that is there,
    and must not overwrite one somebody regenerated on purpose. Absence is the
    only trigger.
    """
    names: list[str] = []

    # The summary is the only agent that searches the web, so it is the only
    # one a missing Tavily key can stop. Extraction and benchmarks read the
    # card and nothing else.
    if doc.summary is None and tavily is not None:
        names.append(SUMMARY)

    checkpoint = doc.checkpoints[0] if doc.checkpoints else None
    if checkpoint is not None:
        if checkpoint.extracted is None:
            names.append(EXTRACTION)
        if checkpoint.extracted_benchmarks is None:
            names.append(BENCHMARKS)

    return names


def enrich_after_first_ingest(
    doc: ModelDoc,
    store: Store,
    card: str | None,
    *,
    llm: LLMSettings | None,
    tavily: TavilySettings | None,
) -> ModelDoc:
    """Run every agent this model has never had, and never fail the ingest.

    This is the one place in the codebase that swallows an error, and it is
    deliberate. Ingest is atomic and owns the document (R1.5); whether an LLM
    answered is not allowed to decide whether a model can enter the vault. The
    failure is not hidden either -- the block comes back absent, which the UI
    renders as "not generated yet, press the button", the same state as a model
    nobody has asked about.

    The card is passed in because ingest already holds it; fetching it again
    would pay for a second request to answer a question already answered.
    """
    if not card or llm is None:
        return doc

    names = _outstanding(doc, tavily)
    if not names:
        return doc

    try:
        for event in run_agents(
            names,
            doc,
            card,
            llm=llm,
            tavily=tavily,
            store=store,
            card_revision=doc.checkpoints[0].card_revision if doc.checkpoints else None,
        ):
            if event.kind == "error":
                log.warning(
                    "%s: %s agent failed on ingest: %s", doc.model_id, event.agent, event.detail
                )
    except Exception as exc:  # noqa: BLE001 - R1.5, the document is already written
        log.warning("could not enrich %s on ingest: %s", doc.model_id, exc)
        return doc

    # The agents wrote through the store; re-read rather than trusting the copy
    # this function was handed.
    return store.read(doc.model_id) or doc
